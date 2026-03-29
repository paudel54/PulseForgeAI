"""
USB Serial data acquisition worker.
Runs in a dedicated QThread and streams ECG/ACC/HR data from custom ESP32
hardware via USB serial at 256000 baud, or from a MockSerialSensor for testing.

The ESP32 firmware outputs CSV lines at 100 Hz with the header:
  seq,tx_ms,rx_ms,latency_ms,inter_arrival_ms,ECG,Resp0,Resp1,SpO2,HR,
  IR,Red,Temp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ

This worker extracts:
  - ECG  (column 5)   → bandpass-filtered downstream at 0.5–25 Hz
  - HR   (column 9)   → emitted at ~1 Hz for the HR plot
  - AccX (column 13)  → triaxial accelerometer
  - AccY (column 14)
  - AccZ (column 15)
"""

import time

from PyQt5.QtCore import QObject, pyqtSignal

from serial_ecg.utils.constants import (
    ECG_NATIVE_HZ, ACC_HZ, SERIAL_BAUD_RATE,
    COL_ECG, COL_HR, COL_ACCX, COL_ACCY, COL_ACCZ,
    MIN_CSV_FIELDS,
)

ECG_BATCH_SIZE = 50    # emit ECG every 50 samples (0.5 s at 100 Hz)
ACC_BATCH_SIZE = 10    # emit ACC every 10 samples (0.1 s at 100 Hz)
HR_EMIT_INTERVAL_S = 1.0


class SerialWorker(QObject):
    """
    Acquires data from custom ESP32 hardware via USB serial.
    Emits Qt signals with the same interface as BLEWorker so the dashboard
    and processing pipeline can be swapped transparently.
    """

    ecg_data     = pyqtSignal(object)  # (timestamp_ns, [ecg_samples])
    acc_data     = pyqtSignal(object)  # (timestamp_ns, [(x,y,z), ...])
    hr_data      = pyqtSignal(object)  # (timestamp_ns, hr_bpm, rr_ms_approx)
    status       = pyqtSignal(str)
    connected    = pyqtSignal(bool)
    device_found = pyqtSignal(str, str)  # (description, port_name)

    def __init__(self, use_mock: bool = False):
        super().__init__()
        self._use_mock = use_mock
        self._running = False
        self._port_name = None
        self._serial = None

    def set_device_address(self, port_name: str):
        self._port_name = port_name

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    #  Trigger command
    # ------------------------------------------------------------------

    def send_trigger(self):
        """Send TRIG command to the microcontroller to fire the BIOPAC trigger pulse."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b"TRIG\n")
                self.status.emit("Trigger command sent (TRIG)")
            except Exception as e:
                self.status.emit(f"Trigger send error: {e}")
        else:
            self.status.emit("Cannot send trigger — serial port not open")

    # ------------------------------------------------------------------
    #  Port scanning
    # ------------------------------------------------------------------

    def run_scan(self):
        """List available COM / serial ports."""
        if self._use_mock:
            self.device_found.emit("Mock Serial Sensor", "MOCK")
            return

        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                self.status.emit("No serial ports found.")
                return
            for p in ports:
                desc = f"{p.description}"
                self.device_found.emit(desc, p.device)
        except Exception as e:
            self.status.emit(f"Port scan error: {e}")

    # ------------------------------------------------------------------
    #  Main entry (called when the QThread starts)
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        if self._use_mock:
            self._run_mock()
        else:
            self._run_serial()

    # ------------------------------------------------------------------
    #  Mock path
    # ------------------------------------------------------------------

    def _run_mock(self):
        from serial_ecg.utils.mock_sensor import MockSerialSensor

        self.status.emit("Mock serial sensor started (100 Hz)")
        self.connected.emit(True)
        sensor = MockSerialSensor()

        ecg_batch = []
        acc_batch = []
        last_hr_time = time.time()

        while self._running:
            row = sensor.get_sample()

            ecg_batch.append(row['ecg'])
            acc_batch.append((row['acc_x'], row['acc_y'], row['acc_z']))

            if len(ecg_batch) >= ECG_BATCH_SIZE:
                self.ecg_data.emit((time.time_ns(), list(ecg_batch)))
                ecg_batch.clear()

            if len(acc_batch) >= ACC_BATCH_SIZE:
                self.acc_data.emit((time.time_ns(), list(acc_batch)))
                acc_batch.clear()

            now = time.time()
            if now - last_hr_time >= HR_EMIT_INTERVAL_S:
                hr = row['hr']
                rr = 60000.0 / hr if hr > 0 else 0
                self.hr_data.emit((time.time_ns(), hr, rr))
                last_hr_time = now

            time.sleep(1.0 / ECG_NATIVE_HZ)

        self.connected.emit(False)
        self.status.emit("Mock serial sensor stopped")

    # ------------------------------------------------------------------
    #  Real serial path
    # ------------------------------------------------------------------

    def _run_serial(self):
        import serial as pyserial

        if not self._port_name:
            self.status.emit("No serial port selected")
            return

        retry_count = 0
        max_retries = 5

        while self._running and retry_count <= max_retries:
            try:
                self.status.emit(
                    f"Opening {self._port_name} at {SERIAL_BAUD_RATE} baud"
                    + (f" (retry {retry_count})" if retry_count > 0 else "")
                )

                self._serial = pyserial.Serial(
                    port=self._port_name,
                    baudrate=SERIAL_BAUD_RATE,
                    timeout=2.0,
                )

                self._serial.reset_input_buffer()
                self.status.emit(f"Connected to {self._port_name}")
                self.connected.emit(True)
                retry_count = 0

                header_line = self._serial.readline().decode('utf-8', errors='replace').strip()
                if header_line:
                    self.status.emit(f"CSV header: {header_line}")

                self._stream_loop()

            except pyserial.SerialException as e:
                retry_count += 1
                self.status.emit(f"Serial error: {e}")
                self.connected.emit(False)
                if retry_count <= max_retries and self._running:
                    wait = min(2 ** retry_count, 10)
                    self.status.emit(f"Retrying in {wait}s...")
                    time.sleep(wait)

            finally:
                if self._serial and self._serial.is_open:
                    self._serial.close()
                self.connected.emit(False)

        if retry_count > max_retries:
            self.status.emit("Max reconnection attempts reached")

    def _stream_loop(self):
        """Continuous CSV parsing loop. Batches samples and emits signals."""
        ecg_batch = []
        acc_batch = []
        last_hr_time = time.time()
        consecutive_errors = 0

        while self._running:
            try:
                raw = self._serial.readline()
                if not raw:
                    continue

                line = raw.decode('utf-8', errors='replace').strip()
                if not line:
                    continue

                fields = line.split(',')
                if len(fields) < MIN_CSV_FIELDS:
                    continue

                ecg_val = float(fields[COL_ECG])
                acc_x   = float(fields[COL_ACCX])
                acc_y   = float(fields[COL_ACCY])
                acc_z   = float(fields[COL_ACCZ])
                hr_val  = float(fields[COL_HR])

                ecg_batch.append(ecg_val)
                acc_batch.append((acc_x, acc_y, acc_z))

                if len(ecg_batch) >= ECG_BATCH_SIZE:
                    self.ecg_data.emit((time.time_ns(), list(ecg_batch)))
                    ecg_batch.clear()

                if len(acc_batch) >= ACC_BATCH_SIZE:
                    self.acc_data.emit((time.time_ns(), list(acc_batch)))
                    acc_batch.clear()

                now = time.time()
                if now - last_hr_time >= HR_EMIT_INTERVAL_S and hr_val > 0:
                    rr = 60000.0 / hr_val
                    self.hr_data.emit((time.time_ns(), hr_val, rr))
                    last_hr_time = now

                consecutive_errors = 0

            except ValueError:
                consecutive_errors += 1
                if consecutive_errors > 50:
                    self.status.emit("Excessive parse errors — check data format")
                    consecutive_errors = 0

            except (OSError, IOError) as e:
                self.status.emit(f"Serial read error: {e}")
                break

        self.status.emit("Serial stream ended")
