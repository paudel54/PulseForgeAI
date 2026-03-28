"""
Main dashboard UI with real-time ECG/ACC/HR plots, HRV metrics sidebar,
and device connection controls.
"""

import time
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QCheckBox, QSplitter,
    QSizePolicy, QApplication, QStatusBar, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette

import pyqtgraph as pg

from polar_ecg.utils.constants import (
    ECG_NATIVE_HZ, ACC_HZ, DARK_THEME,
    WINDOW_SECONDS_OPTIONS, DEFAULT_WINDOW_SECONDS,
)
from polar_ecg.utils.ring_buffer import RingBuffer
from polar_ecg.workers.ble_worker import BLEWorker
from polar_ecg.workers.processing_worker import ProcessingWorker


def _make_dark_palette() -> QPalette:
    t = DARK_THEME
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(t["background"]))
    pal.setColor(QPalette.WindowText, QColor(t["text"]))
    pal.setColor(QPalette.Base, QColor(t["surface"]))
    pal.setColor(QPalette.AlternateBase, QColor(t["background"]))
    pal.setColor(QPalette.ToolTipBase, QColor(t["surface"]))
    pal.setColor(QPalette.ToolTipText, QColor(t["text"]))
    pal.setColor(QPalette.Text, QColor(t["text"]))
    pal.setColor(QPalette.Button, QColor(t["surface"]))
    pal.setColor(QPalette.ButtonText, QColor(t["text"]))
    pal.setColor(QPalette.BrightText, QColor(t["accent"]))
    pal.setColor(QPalette.Highlight, QColor(t["primary"]))
    pal.setColor(QPalette.HighlightedText, QColor(t["background"]))
    return pal


STYLESHEET = """
QMainWindow {{
    background-color: {bg};
}}
QGroupBox {{
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
    color: {text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}
QPushButton {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {border};
}}
QPushButton:pressed {{
    background-color: {primary};
    color: {bg};
}}
QPushButton:disabled {{
    color: {dim};
    background-color: {bg};
}}
QPushButton#connectBtn {{
    background-color: {primary};
    color: {bg};
    font-weight: bold;
}}
QPushButton#freezeBtn {{
    background-color: {accent};
    color: {bg};
    font-weight: bold;
}}
QComboBox {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    color: {text};
    selection-background-color: {primary};
}}
QLabel {{
    color: {text};
}}
QCheckBox {{
    color: {text};
    spacing: 6px;
}}
QTextEdit {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QStatusBar {{
    background-color: {surface};
    color: {dim};
    border-top: 1px solid {border};
}}
""".format(
    bg=DARK_THEME["background"],
    surface=DARK_THEME["surface"],
    primary=DARK_THEME["primary"],
    secondary=DARK_THEME["secondary"],
    accent=DARK_THEME["accent"],
    text=DARK_THEME["text"],
    dim=DARK_THEME["text_dim"],
    border=DARK_THEME["border"],
)


class MainDashboard(QMainWindow):
    """Primary application window with real-time biosignal visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polar ECG Dashboard")
        self.setMinimumSize(1200, 800)

        self._frozen = False
        self._window_seconds = DEFAULT_WINDOW_SECONDS
        self._connected = False

        # Data buffers (120 seconds rolling, numpy ring buffers)
        buf_sec = 120
        self._ecg_buf = RingBuffer(ECG_NATIVE_HZ * buf_sec)
        self._acc_x_buf = RingBuffer(ACC_HZ * buf_sec)
        self._acc_y_buf = RingBuffer(ACC_HZ * buf_sec)
        self._acc_z_buf = RingBuffer(ACC_HZ * buf_sec)
        self._hr_buf = RingBuffer(buf_sec)

        # Workers (created on connect)
        self._ble_worker = None
        self._ble_thread = None
        self._proc_worker = None
        self._proc_thread = None

        self._build_ui()
        self._apply_theme()
        self._start_plot_timer()

    def _apply_theme(self):
        QApplication.instance().setPalette(_make_dark_palette())
        self.setStyleSheet(STYLESHEET)
        pg.setConfigOptions(
            antialias=False,
            background=DARK_THEME["plot_bg"],
            foreground=DARK_THEME["text"],
        )

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._build_toolbar(plot_layout)
        self._build_plots(plot_layout)
        splitter.addWidget(plot_widget)

        # Right: controls + metrics + log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._build_device_controls(right_layout)
        self._build_hrv_panel(right_layout)
        self._build_log_panel(right_layout)
        right_layout.addStretch()
        splitter.addWidget(right_panel)

        splitter.setSizes([900, 350])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Connect a sensor or use mock mode to begin.")

    def _build_toolbar(self, parent_layout):
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Window:"))
        self._window_combo = QComboBox()
        for s in WINDOW_SECONDS_OPTIONS:
            self._window_combo.addItem(f"{s}s", s)
        idx = WINDOW_SECONDS_OPTIONS.index(DEFAULT_WINDOW_SECONDS)
        self._window_combo.setCurrentIndex(idx)
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        toolbar.addWidget(self._window_combo)

        toolbar.addStretch()

        self._freeze_btn = QPushButton("Freeze")
        self._freeze_btn.setObjectName("freezeBtn")
        self._freeze_btn.setCheckable(True)
        self._freeze_btn.toggled.connect(self._on_freeze_toggled)
        self._freeze_btn.setEnabled(False)
        toolbar.addWidget(self._freeze_btn)

        parent_layout.addLayout(toolbar)

    def _build_plots(self, parent_layout):
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ECG
        self._ecg_plot = self._plot_widget.addPlot(
            row=0, col=0, title="ECG (130 Hz)"
        )
        self._ecg_plot.setLabel("left", "Amplitude", units="uV")
        self._ecg_plot.showGrid(x=True, y=True, alpha=0.15)
        self._ecg_plot.setDownsampling(auto=True, mode="peak")
        self._ecg_plot.setClipToView(True)
        self._ecg_curve = self._ecg_plot.plot(
            pen=pg.mkPen(DARK_THEME["ecg_color"], width=1.5)
        )

        # Accelerometer
        self._acc_plot = self._plot_widget.addPlot(
            row=1, col=0, title="Accelerometer (100 Hz)"
        )
        self._acc_plot.setLabel("left", "Acceleration", units="mg")
        self._acc_plot.showGrid(x=True, y=True, alpha=0.15)
        self._acc_plot.setDownsampling(auto=True, mode="peak")
        self._acc_plot.setClipToView(True)
        self._acc_plot.setXLink(self._ecg_plot)
        self._acc_plot.addLegend(offset=(10, 10))
        self._acc_x_curve = self._acc_plot.plot(
            pen=pg.mkPen(DARK_THEME["acc_x_color"], width=1.2), name="X"
        )
        self._acc_y_curve = self._acc_plot.plot(
            pen=pg.mkPen(DARK_THEME["acc_y_color"], width=1.2), name="Y"
        )
        self._acc_z_curve = self._acc_plot.plot(
            pen=pg.mkPen(DARK_THEME["acc_z_color"], width=1.2), name="Z"
        )

        # Heart Rate
        self._hr_plot = self._plot_widget.addPlot(row=2, col=0, title="Heart Rate")
        self._hr_plot.setLabel("left", "BPM")
        self._hr_plot.setLabel("bottom", "Time", units="s")
        self._hr_plot.showGrid(x=True, y=True, alpha=0.15)
        self._hr_plot.setXLink(self._ecg_plot)
        self._hr_curve = self._hr_plot.plot(
            pen=pg.mkPen(DARK_THEME["hr_color"], width=2),
            symbol="o", symbolSize=5,
            symbolBrush=DARK_THEME["hr_color"],
        )

        parent_layout.addWidget(self._plot_widget)

    def _build_device_controls(self, parent_layout):
        group = QGroupBox("Device Connection")
        layout = QVBoxLayout()

        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self._on_scan)
        scan_row.addWidget(self._scan_btn)

        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(180)
        self._device_combo.addItem("No devices found")
        scan_row.addWidget(self._device_combo)
        layout.addLayout(scan_row)

        connect_row = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("connectBtn")
        self._connect_btn.clicked.connect(self._on_connect)
        connect_row.addWidget(self._connect_btn)

        self._mock_btn = QPushButton("Mock Sensor")
        self._mock_btn.clicked.connect(self._on_mock_connect)
        connect_row.addWidget(self._mock_btn)

        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        connect_row.addWidget(self._disconnect_btn)
        layout.addLayout(connect_row)

        self._conn_label = QLabel("Status: Disconnected")
        self._conn_label.setStyleSheet(f"color: {DARK_THEME['accent']};")
        layout.addWidget(self._conn_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_hrv_panel(self, parent_layout):
        group = QGroupBox("HRV Analysis (Rolling)")
        layout = QGridLayout()

        self._hrv_enabled_cb = QCheckBox("Enable HRV Analysis")
        self._hrv_enabled_cb.setChecked(True)
        self._hrv_enabled_cb.toggled.connect(self._on_hrv_toggle)
        layout.addWidget(self._hrv_enabled_cb, 0, 0, 1, 2)

        hrv_fields = [
            ("RMSSD:", "rmssd"),
            ("SDNN:", "sdnn"),
            ("LF/HF:", "lf_hf"),
            ("Mean HR:", "mean_hr"),
            ("P Width:", "p_width"),
            ("QRS Width:", "qrs_width"),
            ("ST Segment:", "st_width"),
            ("QT Interval:", "qt_width"),
            ("QTc (Bazett):", "qtc_width"),
        ]
        self._hrv_labels = {}
        for i, (display, key) in enumerate(hrv_fields):
            lbl = QLabel(display)
            lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            layout.addWidget(lbl, i + 1, 0)
            val = QLabel("--")
            val.setFont(QFont("Consolas", 11))
            val.setStyleSheet(f"color: {DARK_THEME['secondary']};")
            layout.addWidget(val, i + 1, 1)
            self._hrv_labels[key] = val

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_log_panel(self, parent_layout):
        group = QGroupBox("Log")
        layout = QVBoxLayout()
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(160)
        layout.addWidget(self._log_text)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{ts}] {msg}")
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )

    # ------------------------------------------------------------------ #
    #  Plot updates (~30 FPS)
    # ------------------------------------------------------------------ #

    def _start_plot_timer(self):
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(33)

    def _update_plots(self):
        if self._frozen:
            return

        win = self._window_seconds

        n_ecg = int(win * ECG_NATIVE_HZ)
        ecg_data = self._ecg_buf.get_last_n(n_ecg)
        if len(ecg_data) > 0:
            t_ecg = np.arange(len(ecg_data), dtype=np.float32) * (1.0 / ECG_NATIVE_HZ)
            self._ecg_curve.setData(t_ecg, ecg_data)

        n_acc = int(win * ACC_HZ)
        ax = self._acc_x_buf.get_last_n(n_acc)
        ay = self._acc_y_buf.get_last_n(n_acc)
        az = self._acc_z_buf.get_last_n(n_acc)
        if len(ax) > 0:
            t_acc = np.arange(len(ax), dtype=np.float32) * (1.0 / ACC_HZ)
            self._acc_x_curve.setData(t_acc, ax)
            self._acc_y_curve.setData(t_acc, ay)
            self._acc_z_curve.setData(t_acc, az)

        hr_data = self._hr_buf.get_last_n(win)
        if len(hr_data) > 0:
            t_hr = np.arange(len(hr_data), dtype=np.float32)
            self._hr_curve.setData(t_hr, hr_data)

    # ------------------------------------------------------------------ #
    #  Device connection
    # ------------------------------------------------------------------ #

    def _on_scan(self):
        self._device_combo.clear()
        self._scan_btn.setEnabled(False)
        self._log("Scanning for Polar devices...")

        worker = BLEWorker(use_mock=False)
        worker.device_found.connect(self._on_device_found)
        worker.status.connect(self._log)

        self._scan_thread = QThread()
        worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(worker.run_scan)
        self._scan_thread.finished.connect(lambda: self._scan_btn.setEnabled(True))

        def _finish_scan(msg):
            if self._scan_thread.isRunning():
                self._scan_thread.quit()

        worker.status.connect(_finish_scan)
        self._scan_worker = worker
        self._scan_thread.start()

    @pyqtSlot(str, str)
    def _on_device_found(self, name, address):
        self._device_combo.addItem(f"{name} ({address})", address)
        self._log(f"Found: {name} [{address}]")

    def _on_connect(self):
        address = self._device_combo.currentData()
        if not address:
            self._log("No device selected")
            return
        self._start_acquisition(use_mock=False, address=address)

    def _on_mock_connect(self):
        self._start_acquisition(use_mock=True)

    def _start_acquisition(self, use_mock: bool, address: str = None):
        if self._ble_thread and self._ble_thread.isRunning():
            self._log("Already connected")
            return

        self._ble_worker = BLEWorker(use_mock=use_mock)
        if address:
            self._ble_worker.set_device_address(address)

        self._ble_thread = QThread()
        self._ble_worker.moveToThread(self._ble_thread)
        self._ble_thread.started.connect(self._ble_worker.run)

        self._ble_worker.ecg_data.connect(self._on_ecg_data)
        self._ble_worker.acc_data.connect(self._on_acc_data)
        self._ble_worker.hr_data.connect(self._on_hr_data)
        self._ble_worker.status.connect(self._log)
        self._ble_worker.connected.connect(self._on_connection_changed)

        self._proc_worker = ProcessingWorker()
        self._proc_thread = QThread()
        self._proc_worker.moveToThread(self._proc_thread)
        self._proc_thread.started.connect(self._proc_worker.run)

        self._proc_worker.hrv_result.connect(self._on_hrv_result)
        self._proc_worker.status.connect(self._log)

        self._proc_thread.start()
        self._ble_thread.start()

        self._connect_btn.setEnabled(False)
        self._mock_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._freeze_btn.setEnabled(True)

    def _on_disconnect(self):
        if self._ble_worker:
            self._ble_worker.stop()
        if self._proc_worker:
            self._proc_worker.stop()

        if self._ble_thread:
            self._ble_thread.quit()
            self._ble_thread.wait(3000)
        if self._proc_thread:
            self._proc_thread.quit()
            self._proc_thread.wait(3000)

        self._ble_worker = None
        self._proc_worker = None

        self._connect_btn.setEnabled(True)
        self._mock_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._freeze_btn.setEnabled(False)

        self._on_connection_changed(False)
        self._log("Disconnected")

    @pyqtSlot(bool)
    def _on_connection_changed(self, connected):
        self._connected = connected
        if connected:
            self._conn_label.setText("Status: Connected")
            self._conn_label.setStyleSheet(f"color: {DARK_THEME['secondary']};")
            self._status_bar.showMessage("Streaming data...")
        else:
            self._conn_label.setText("Status: Disconnected")
            self._conn_label.setStyleSheet(f"color: {DARK_THEME['accent']};")
            self._status_bar.showMessage("Disconnected")

    # ------------------------------------------------------------------ #
    #  Data handlers
    # ------------------------------------------------------------------ #

    @pyqtSlot(object)
    def _on_ecg_data(self, data):
        ts, samples = data
        self._ecg_buf.extend(samples)
        if self._proc_worker:
            self._proc_worker.add_raw_ecg(samples)

    @pyqtSlot(object)
    def _on_acc_data(self, data):
        ts, samples = data
        if samples:
            arr = np.array(samples, dtype=np.float64)
            self._acc_x_buf.extend(arr[:, 0])
            self._acc_y_buf.extend(arr[:, 1])
            self._acc_z_buf.extend(arr[:, 2])

    @pyqtSlot(object)
    def _on_hr_data(self, data):
        ts, hr, rr = data
        self._hr_buf.append(hr)

    # ------------------------------------------------------------------ #
    #  Toolbar actions
    # ------------------------------------------------------------------ #

    def _on_window_changed(self, idx):
        self._window_seconds = self._window_combo.itemData(idx)

    def _on_freeze_toggled(self, checked):
        self._frozen = checked
        self._freeze_btn.setText("Resume" if checked else "Freeze")

    def _on_hrv_toggle(self, enabled):
        if self._proc_worker:
            self._proc_worker.set_hrv_enabled(enabled)

    @pyqtSlot(object)
    def _on_hrv_result(self, result: dict):
        for key in self._hrv_labels:
            val = result.get(key)
            if val is not None:
                if key == "mean_hr":
                    text = f"{val:.0f} bpm"
                elif key == "lf_hf":
                    text = f"{val:.3f}"
                else:
                    text = f"{val:.1f} ms"
                self._hrv_labels[key].setText(text)
            else:
                self._hrv_labels[key].setText("--")

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        self._on_disconnect()
        event.accept()
