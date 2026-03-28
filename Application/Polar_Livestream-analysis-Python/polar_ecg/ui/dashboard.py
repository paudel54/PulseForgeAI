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
    QSpinBox, QDoubleSpinBox, QTabWidget,
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
from polar_ecg.utils.vo2max_estimator import VO2MaxEstimator
from polar_ecg.utils.nn_vo2max import CardioFitnessNN


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
QComboBox, QSpinBox, QDoubleSpinBox {{
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
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {border};
    border: none;
    border-radius: 2px;
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
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 4px;
    background-color: {surface};
}}
QTabBar::tab {{
    background-color: {bg};
    color: {dim};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    background-color: {surface};
    color: {text};
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{
    background-color: {border};
    color: {text};
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
        self._last_hrv_result: dict = {}
        self._vo2max_est = VO2MaxEstimator(age=30, sex="male", weight_kg=70.0)
        self._nn = CardioFitnessNN()
        # Accumulate ACC samples → 1-Hz epochs for the NN
        self._acc_epoch_accum: list = []
        self._acc_epoch_count = 0

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
        self._build_user_profile_panel(right_layout)
        self._build_hrv_panel(right_layout)
        self._build_fitness_panel(right_layout)
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

    def _build_user_profile_panel(self, parent_layout):
        group = QGroupBox("User Profile")
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        def _lbl(t):
            l = QLabel(t)
            l.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
            return l

        grid.addWidget(_lbl("Age"),  0, 0)
        self._age_spin = QSpinBox()
        self._age_spin.setRange(13, 100)
        self._age_spin.setValue(30)
        self._age_spin.setSuffix(" yr")
        self._age_spin.valueChanged.connect(self._on_profile_changed)
        grid.addWidget(self._age_spin, 0, 1)

        grid.addWidget(_lbl("Sex"),  0, 2)
        self._sex_combo = QComboBox()
        self._sex_combo.addItems(["Male", "Female"])
        self._sex_combo.currentIndexChanged.connect(self._on_profile_changed)
        grid.addWidget(self._sex_combo, 0, 3)

        grid.addWidget(_lbl("Weight"),  1, 0)
        self._weight_spin = QDoubleSpinBox()
        self._weight_spin.setRange(30.0, 250.0)
        self._weight_spin.setValue(70.0)
        self._weight_spin.setSuffix(" kg")
        self._weight_spin.setSingleStep(0.5)
        self._weight_spin.valueChanged.connect(self._on_profile_changed)
        grid.addWidget(self._weight_spin, 1, 1)

        grid.addWidget(_lbl("Height"),  1, 2)
        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(1.20, 2.50)
        self._height_spin.setValue(1.75)
        self._height_spin.setSuffix(" m")
        self._height_spin.setSingleStep(0.01)
        self._height_spin.setDecimals(2)
        self._height_spin.valueChanged.connect(self._on_profile_changed)
        grid.addWidget(self._height_spin, 1, 3)

        group.setLayout(grid)
        parent_layout.addWidget(group)

    def _build_fitness_panel(self, parent_layout):
        group = QGroupBox("VO₂max Estimate")
        layout = QGridLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setVerticalSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        T = DARK_THEME

        def _hdr(text, align=Qt.AlignLeft):
            lbl = QLabel(text)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            lbl.setStyleSheet(f"color: {T['text_dim']};")
            lbl.setAlignment(align)
            return lbl

        def _row_lbl(text):
            lbl = QLabel(text)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            return lbl

        def _val(text="--", size=11, color=None, bold=False):
            lbl = QLabel(text)
            w = QFont.Bold if bold else QFont.Normal
            lbl.setFont(QFont("Consolas", size, w))
            lbl.setStyleSheet(f"color: {color or T['secondary']};")
            return lbl

        # Column headers
        layout.addWidget(_hdr(""), 0, 0)
        layout.addWidget(_hdr("Formula", Qt.AlignHCenter), 0, 1)
        layout.addWidget(_hdr("Neural Net", Qt.AlignHCenter), 0, 2)

        # Separator-style top
        r = 1

        # VO2max row
        layout.addWidget(_row_lbl("VO₂max"), r, 0)
        self._vo2max_val = _val(size=12, bold=True)
        self._vo2max_val.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._vo2max_val, r, 1)
        self._nn_vo2max_val = _val(size=12, bold=True, color=T["primary"])
        self._nn_vo2max_val.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._nn_vo2max_val, r, 2)
        r += 1

        # Category row
        layout.addWidget(_row_lbl("Category"), r, 0)
        self._fitness_cat_lbl = _val(size=10, bold=True)
        self._fitness_cat_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._fitness_cat_lbl, r, 1)
        self._nn_cat_lbl = _val(size=10, bold=True, color=T["primary"])
        self._nn_cat_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._nn_cat_lbl, r, 2)
        r += 1

        # Status / method row
        layout.addWidget(_row_lbl("Method"), r, 0)
        self._vo2max_method_lbl = _val(size=9, color=T["text_dim"])
        self._vo2max_method_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._vo2max_method_lbl, r, 1)
        self._nn_status_lbl = _val(size=9, color=T["text_dim"])
        self._nn_status_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._nn_status_lbl, r, 2)
        r += 1

        # HR info row
        layout.addWidget(_row_lbl("Resting HR"), r, 0)
        self._rest_hr_lbl = _val(size=10)
        self._rest_hr_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._rest_hr_lbl, r, 1)
        self._max_hr_lbl = _val(size=10, color=T["text_dim"])
        self._max_hr_lbl.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._max_hr_lbl, r, 2)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_hrv_panel(self, parent_layout):
        group = QGroupBox("Analysis")
        outer = QVBoxLayout()
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self._hrv_enabled_cb = QCheckBox("Enable HRV Analysis")
        self._hrv_enabled_cb.setChecked(True)
        self._hrv_enabled_cb.toggled.connect(self._on_hrv_toggle)
        outer.addWidget(self._hrv_enabled_cb)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # ---- Tab 1: HRV metrics ----
        hrv_tab = QWidget()
        hrv_grid = QGridLayout(hrv_tab)
        hrv_grid.setContentsMargins(8, 10, 8, 8)
        hrv_grid.setHorizontalSpacing(16)
        hrv_grid.setVerticalSpacing(2)

        self._hrv_labels = {}
        hrv_fields = [
            ("RMSSD", "rmssd",   "ms"),
            ("SDNN",  "sdnn",    "ms"),
            ("LF/HF", "lf_hf",  ""),
            ("HR",    "mean_hr", "bpm"),
        ]
        for i, (label, key, _unit) in enumerate(hrv_fields):
            col = (i % 2) * 2
            row = i // 2
            hdr = QLabel(label)
            hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
            hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
            hrv_grid.addWidget(hdr, row * 2, col)
            val = QLabel("--")
            val.setFont(QFont("Consolas", 11))
            val.setStyleSheet(f"color: {DARK_THEME['secondary']};")
            hrv_grid.addWidget(val, row * 2 + 1, col)
            self._hrv_labels[key] = val

        tabs.addTab(hrv_tab, "HRV")

        # ---- Tab 2: ECG morphology ----
        ecg_tab = QWidget()
        ecg_grid = QGridLayout(ecg_tab)
        ecg_grid.setContentsMargins(8, 10, 8, 8)
        ecg_grid.setHorizontalSpacing(16)
        ecg_grid.setVerticalSpacing(2)

        self._ecg_labels = {}
        ecg_fields = [
            ("Avg HR",  "mean_hr",   "bpm"),
            ("QRS",     "qrs_width", "ms"),
            ("ST seg",  "st_width",  "ms"),
            ("QT",      "qt_width",  "ms"),
            ("QTc",     "qtc_width", "ms"),
            ("P width", "p_width",   "ms"),
        ]
        for i, (label, key, _unit) in enumerate(ecg_fields):
            col = (i % 2) * 2
            row = i // 2
            hdr = QLabel(label)
            hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
            hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
            ecg_grid.addWidget(hdr, row * 2, col)
            val = QLabel("--")
            val.setFont(QFont("Consolas", 11))
            val.setStyleSheet(f"color: {DARK_THEME['secondary']};")
            ecg_grid.addWidget(val, row * 2 + 1, col)
            self._ecg_labels[key] = val

        tabs.addTab(ecg_tab, "ECG")

        outer.addWidget(tabs)
        group.setLayout(outer)
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
            # Compute per-sample vector magnitude and accumulate into 1-Hz epochs
            mag = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
            self._acc_epoch_accum.extend(mag.tolist())
            self._acc_epoch_count += len(mag)
            if self._acc_epoch_count >= ACC_HZ:  # ~1 second
                epoch_mag = float(np.mean(self._acc_epoch_accum))
                self._nn.add_acc_mag_epoch(epoch_mag)
                self._acc_epoch_accum.clear()
                self._acc_epoch_count = 0

    @pyqtSlot(object)
    def _on_hr_data(self, data):
        ts, hr, rr = data
        self._hr_buf.append(hr)
        self._vo2max_est.update_hr(hr)
        self._nn.add_hr(hr)

    # ------------------------------------------------------------------ #
    #  Toolbar actions
    # ------------------------------------------------------------------ #

    def _on_profile_changed(self):
        age    = self._age_spin.value()
        sex    = "male" if self._sex_combo.currentIndex() == 0 else "female"
        weight = self._weight_spin.value()
        height = self._height_spin.value()
        self._vo2max_est.update_profile(age, sex, weight)
        self._nn.update_profile(age, sex, height, weight)
        if self._last_hrv_result:
            self._update_fitness_display(self._last_hrv_result)

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
        self._last_hrv_result = result

        # HRV tab
        _hrv_fmt = {
            "rmssd":   lambda v: f"{v:.1f} ms",
            "sdnn":    lambda v: f"{v:.1f} ms",
            "lf_hf":   lambda v: f"{v:.2f}",
            "mean_hr": lambda v: f"{v:.0f} bpm",
        }
        for key, lbl in self._hrv_labels.items():
            val = result.get(key)
            lbl.setText(_hrv_fmt[key](val) if val is not None else "--")

        # ECG tab
        _ecg_fmt = {
            "mean_hr":   lambda v: f"{v:.0f} bpm",
            "qrs_width": lambda v: f"{v:.1f} ms",
            "st_width":  lambda v: f"{v:.1f} ms",
            "qt_width":  lambda v: f"{v:.1f} ms",
            "qtc_width": lambda v: f"{v:.1f} ms",
            "p_width":   lambda v: f"{v:.1f} ms",
        }
        for key, lbl in self._ecg_labels.items():
            val = result.get(key)
            lbl.setText(_ecg_fmt[key](val) if val is not None else "--")

        rmssd = result.get("rmssd")
        if rmssd is not None:
            self._nn.add_hrv_rmssd(rmssd)

        self._update_fitness_display(result)

    def _update_fitness_display(self, hrv_result: dict):
        """Render the two-column VO2max comparison (Formula | Neural Net)."""
        rmssd   = hrv_result.get("rmssd")
        fitness = self._vo2max_est.best_estimate(rmssd=rmssd)

        # --- Formula column ---
        if fitness["vo2max"] is not None:
            self._vo2max_val.setText(f"{fitness['vo2max']:.1f}")
        else:
            self._vo2max_val.setText("--")

        self._vo2max_method_lbl.setText(fitness["method"])

        if fitness["category"]:
            self._fitness_cat_lbl.setText(fitness["category"])
            self._fitness_cat_lbl.setStyleSheet(
                f"color: {fitness['category_color']}; font-weight: bold;"
            )
        else:
            self._fitness_cat_lbl.setText("--")
            self._fitness_cat_lbl.setStyleSheet(
                f"color: {DARK_THEME['text_dim']}; font-weight: bold;"
            )

        hr_rest = fitness["hr_rest"]
        self._rest_hr_lbl.setText(
            f"{hr_rest:.0f} bpm" if hr_rest is not None else "--"
        )
        self._max_hr_lbl.setText(f"max {fitness['hr_max']:.0f} bpm")

        # --- Neural Net column ---
        nn_result = self._nn.predict()
        nn_vo2 = nn_result.get("vo2max")
        nn_status = nn_result.get("status", "")

        if nn_vo2 is not None:
            self._nn_vo2max_val.setText(f"{nn_vo2:.1f}")
            nn_cat = self._vo2max_est._get_fitness_category(nn_vo2)
            from polar_ecg.utils.vo2max_estimator import CATEGORY_COLORS
            self._nn_cat_lbl.setText(nn_cat)
            self._nn_cat_lbl.setStyleSheet(
                f"color: {CATEGORY_COLORS.get(nn_cat, DARK_THEME['primary'])}; font-weight: bold;"
            )
            self._nn_status_lbl.setText("CardioFitness NN")
        else:
            self._nn_vo2max_val.setText("--")
            self._nn_cat_lbl.setText("--")
            self._nn_cat_lbl.setStyleSheet(
                f"color: {DARK_THEME['text_dim']}; font-weight: bold;"
            )
            self._nn_status_lbl.setText(nn_status)

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        self._on_disconnect()
        event.accept()
