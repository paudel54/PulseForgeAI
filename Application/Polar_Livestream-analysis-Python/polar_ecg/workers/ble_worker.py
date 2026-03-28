"""
BLE data acquisition worker.
Runs in a dedicated QThread and streams ECG/ACC/HR data from Polar H10
via bleakheart, or from a MockPolarH10 for testing.
"""

import asyncio
import time

from PyQt5.QtCore import QObject, pyqtSignal

from polar_ecg.utils.constants import ECG_NATIVE_HZ, ACC_HZ
from polar_ecg.utils.mock_sensor import MockPolarH10

# Polar PMD: HR often appears before ECG/ACC; wait for real HR, prime PMD, then verify frames.
HR_FIRST_PACKET_TIMEOUT_S = 25.0
PMD_FIRST_FRAME_TIMEOUT_ECG_S = 8.0
PMD_FIRST_FRAME_TIMEOUT_ACC_S = 7.0
PMD_STREAM_ATTEMPTS = 4


class BLEWorker(QObject):
    """
    Acquires data from a Polar H10 via BLE or mock sensor.
    Emits Qt signals whenever new data frames arrive.
    """

    ecg_data = pyqtSignal(object)    # (timestamp_ns, [samples_uV])
    acc_data = pyqtSignal(object)    # (timestamp_ns, [(x,y,z), ...])
    hr_data = pyqtSignal(object)     # (timestamp_ns, hr_bpm, rr_ms)
    status = pyqtSignal(str)         # status messages
    connected = pyqtSignal(bool)     # connection state changes
    device_found = pyqtSignal(str, str)  # (name, address)

    def __init__(self, use_mock: bool = False):
        super().__init__()
        self._use_mock = use_mock
        self._running = False
        self._device_address = None
        self._loop = None

    def set_device_address(self, address: str):
        self._device_address = address

    def stop(self):
        self._running = False

    def run_scan(self):
        """Scan for available Polar devices."""
        if self._use_mock:
            self.device_found.emit("Mock Polar H10", "00:00:00:00:00:00")
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_scan())
            loop.close()
        except Exception as e:
            self.status.emit(f"Scan error: {e}")

    async def _async_scan(self):
        from bleak import BleakScanner

        self.status.emit("Scanning for Polar devices...")

        devices = await BleakScanner.discover(timeout=8.0)
        found = False
        for d in devices:
            name = d.name or ""
            if "polar" in name.lower():
                self.device_found.emit(name, d.address)
                found = True

        if not found:
            self.status.emit("No Polar devices found. Ensure sensor is on and nearby.")

    def run(self):
        """Main acquisition loop. Called when the thread starts."""
        self._running = True

        if self._use_mock:
            self._run_mock()
        else:
            self._run_ble()

    def _run_mock(self):
        """Generate synthetic data for UI testing."""
        self.status.emit("Mock sensor started")
        self.connected.emit(True)
        sensor = MockPolarH10()

        ecg_interval = 73.0 / ECG_NATIVE_HZ  # ~0.56s per frame
        acc_interval = 16.0 / ACC_HZ          # ~0.16s per frame
        hr_interval = 1.0

        last_ecg = time.time()
        last_acc = time.time()
        last_hr = time.time()

        while self._running:
            now = time.time()

            if now - last_ecg >= ecg_interval:
                tag, ts, samples = sensor.get_ecg_frame()
                self.ecg_data.emit((ts, samples))
                last_ecg = now

            if now - last_acc >= acc_interval:
                tag, ts, samples = sensor.get_acc_frame()
                self.acc_data.emit((ts, samples))
                last_acc = now

            if now - last_hr >= hr_interval:
                tag, ts, (hr, rr), _ = sensor.get_hr_frame()
                self.hr_data.emit((ts, hr, rr))
                last_hr = now

            time.sleep(0.005)

        self.connected.emit(False)
        self.status.emit("Mock sensor stopped")

    def _run_ble(self):
        """Connect to a real Polar H10 and stream data."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_stream())
        except Exception as e:
            self.status.emit(f"BLE error: {e}")
            self.connected.emit(False)
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _wait_first_hr_packet(self, hr_queue: asyncio.Queue, timeout: float) -> bool:
        """Block until the first HR notification or timeout. Emits that packet to the UI."""
        try:
            pkt = await asyncio.wait_for(hr_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        _tag, ts, hr_rr, _energy = pkt
        hr, rr = hr_rr
        self.hr_data.emit((ts, hr, rr))
        return True

    async def _negotiate_mtu(self, client) -> None:
        """Request a larger ATT MTU when the platform supports it (helps full ECG frames)."""
        try:
            exch = getattr(client, "exchange_mtu", None)
            if callable(exch):
                await exch(247)
                self.status.emit("BLE MTU negotiated (247)")
        except Exception as e:
            self.status.emit(f"BLE MTU negotiation skipped: {e}")

    async def _prime_pmd(self, pmd, meas_list: list) -> None:
        """
        Start PMD control/data notifications via a capability read before START commands.
        Uses bleakheart public API (first _pmd_ctrl_request starts notifications).
        """
        prime = "ECG" if "ECG" in meas_list else ("ACC" if "ACC" in meas_list else None)
        if not prime:
            return
        info = await pmd.available_settings(prime)
        ec = info.get("error_code", -1)
        if ec != 0:
            self.status.emit(
                f"PMD priming ({prime}) non-success: {info.get('error_msg', info)}"
            )
        else:
            self.status.emit(f"PMD control channel ready ({prime} settings read OK)")

    async def _start_pmd_until_first_frame(
        self,
        pmd,
        measurement: str,
        stream_kwargs: dict,
        data_queue: asyncio.Queue,
        frame_timeout: float,
        attempts: int,
    ) -> bool:
        """
        START streaming, wait for one decoded frame, emit it, return True.
        On timeout, STOP and retry (handles silent Polar PMD failures on Windows).
        """
        for attempt in range(1, attempts + 1):
            err, msg, _ = await pmd.start_streaming(measurement, **stream_kwargs)
            if err != 0:
                self.status.emit(
                    f"{measurement} START error {err}: {msg} (attempt {attempt}/{attempts})"
                )
                await asyncio.sleep(0.7)
                continue
            try:
                tag, ts, payload = await asyncio.wait_for(
                    data_queue.get(), timeout=frame_timeout
                )
            except asyncio.TimeoutError:
                self.status.emit(
                    f"{measurement}: no data within {frame_timeout:.0f}s "
                    f"(attempt {attempt}/{attempts}); stopping stream and retrying"
                )
                try:
                    await pmd.stop_streaming(measurement)
                except Exception:
                    pass
                await asyncio.sleep(0.8)
                continue

            if measurement == "ECG":
                self.ecg_data.emit((ts, payload))
            else:
                self.acc_data.emit((ts, payload))
            return True
        return False

    async def _async_stream(self):
        import bleakheart as bh
        from bleak import BleakClient

        if not self._device_address:
            self.status.emit("No device address set")
            return

        disconnected_event = asyncio.Event()

        def on_disconnect(client):
            self.status.emit("Device disconnected")
            self.connected.emit(False)
            disconnected_event.set()

        retry_count = 0
        max_retries = 5

        while self._running and retry_count <= max_retries:
            try:
                self.status.emit(
                    f"Connecting to {self._device_address}..."
                    + (f" (retry {retry_count})" if retry_count > 0 else "")
                )

                async with BleakClient(
                    self._device_address,
                    disconnected_callback=on_disconnect,
                    timeout=20.0,
                ) as client:
                    self.status.emit("Connected. Starting data streams...")
                    self.connected.emit(True)
                    retry_count = 0
                    disconnected_event.clear()

                    ecg_queue = asyncio.Queue()
                    acc_queue = asyncio.Queue()
                    hr_queue = asyncio.Queue()

                    heartrate = bh.HeartRate(
                        client, queue=hr_queue, unpack=True, instant_rate=True
                    )
                    pmd = bh.PolarMeasurementData(
                        client, ecg_queue=ecg_queue, acc_queue=acc_queue
                    )

                    await heartrate.start_notify()
                    got_hr = await self._wait_first_hr_packet(
                        hr_queue, HR_FIRST_PACKET_TIMEOUT_S
                    )
                    if not got_hr:
                        self.status.emit(
                            "No HR packet yet (strap on / wet electrodes?). "
                            "Continuing with PMD setup anyway."
                        )

                    await self._negotiate_mtu(client)

                    meas = await pmd.available_measurements()
                    self.status.emit(f"Available measurements: {meas}")

                    await self._prime_pmd(pmd, meas)

                    ecg_ok = False
                    if "ECG" in meas:
                        ecg_ok = await self._start_pmd_until_first_frame(
                            pmd,
                            "ECG",
                            {},
                            ecg_queue,
                            PMD_FIRST_FRAME_TIMEOUT_ECG_S,
                            PMD_STREAM_ATTEMPTS,
                        )
                        if not ecg_ok:
                            self.status.emit(
                                "ECG: could not confirm data after several tries"
                            )
                    else:
                        self.status.emit("ECG not available on this device")

                    acc_ok = False
                    if "ACC" in meas:
                        acc_ok = await self._start_pmd_until_first_frame(
                            pmd,
                            "ACC",
                            {"SAMPLE_RATE": 100},
                            acc_queue,
                            PMD_FIRST_FRAME_TIMEOUT_ACC_S,
                            PMD_STREAM_ATTEMPTS,
                        )
                        if not acc_ok:
                            self.status.emit(
                                "ACC: could not confirm data after several tries"
                            )
                    else:
                        self.status.emit("ACC not available on this device")

                    if ecg_ok and acc_ok:
                        self.status.emit("ECG and ACC streaming (first frames received)")
                    elif ecg_ok or acc_ok:
                        self.status.emit("Partial PMD streaming (see messages above)")
                    elif "ECG" in meas or "ACC" in meas:
                        self.status.emit(
                            "PMD did not deliver ECG/ACC this session; try Disconnect "
                            "and Connect, or move the sensor closer to the radio."
                        )

                    while self._running and not disconnected_event.is_set():
                        await self._drain_queues(
                            ecg_queue, acc_queue, hr_queue
                        )
                        await asyncio.sleep(0.01)

                    try:
                        await heartrate.stop_notify()
                        await pmd.stop_streaming("ECG")
                        await pmd.stop_streaming("ACC")
                    except Exception:
                        pass

            except Exception as e:
                retry_count += 1
                self.status.emit(f"Connection lost: {e}")
                self.connected.emit(False)
                if retry_count <= max_retries and self._running:
                    wait = min(2 ** retry_count, 10)
                    self.status.emit(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)

        if retry_count > max_retries:
            self.status.emit("Max reconnection attempts reached")

    async def _drain_queues(self, ecg_q, acc_q, hr_q):
        """Pull all available frames from the async queues and emit signals."""
        while not ecg_q.empty():
            try:
                tag, ts, samples = ecg_q.get_nowait()
                self.ecg_data.emit((ts, samples))
            except asyncio.QueueEmpty:
                break

        while not acc_q.empty():
            try:
                tag, ts, samples = acc_q.get_nowait()
                self.acc_data.emit((ts, samples))
            except asyncio.QueueEmpty:
                break

        while not hr_q.empty():
            try:
                tag, ts, (hr, rr), energy = hr_q.get_nowait()
                self.hr_data.emit((ts, hr, rr))
            except asyncio.QueueEmpty:
                break
