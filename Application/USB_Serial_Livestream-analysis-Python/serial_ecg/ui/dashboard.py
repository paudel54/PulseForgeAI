"""
Main dashboard UI -- real-time ECG/ACC/HR plots, HRV metrics sidebar,
subject recording controls, trigger button, and 5-second JSON data export.

This version acquires data via USB serial from custom ESP32 hardware
instead of BLE from a Polar H10.
"""

import time

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QCheckBox, QSplitter,
    QSizePolicy, QApplication, QStatusBar, QGridLayout,
    QSpinBox, QDoubleSpinBox, QTabWidget, QLineEdit,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette

import pyqtgraph as pg

from serial_ecg.utils.constants import (
    ECG_NATIVE_HZ, ACC_HZ, DARK_THEME,
    WINDOW_SECONDS_OPTIONS, DEFAULT_WINDOW_SECONDS,
    SERIAL_BAUD_RATE,
)
from serial_ecg.utils.ring_buffer import RingBuffer
from serial_ecg.utils.data_exporter import DataExporter
from serial_ecg.workers.serial_worker import SerialWorker
from serial_ecg.workers.processing_worker import ProcessingWorker
from serial_ecg.workers.mqtt_worker import MQTTWorker


def _make_dark_palette() -> QPalette:
    t = DARK_THEME
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(t["background"]))
    pal.setColor(QPalette.WindowText,      QColor(t["text"]))
    pal.setColor(QPalette.Base,            QColor(t["surface"]))
    pal.setColor(QPalette.AlternateBase,   QColor(t["background"]))
    pal.setColor(QPalette.ToolTipBase,     QColor(t["surface"]))
    pal.setColor(QPalette.ToolTipText,     QColor(t["text"]))
    pal.setColor(QPalette.Text,            QColor(t["text"]))
    pal.setColor(QPalette.Button,          QColor(t["surface"]))
    pal.setColor(QPalette.ButtonText,      QColor(t["text"]))
    pal.setColor(QPalette.BrightText,      QColor(t["accent"]))
    pal.setColor(QPalette.Highlight,       QColor(t["primary"]))
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
QPushButton#recordBtn {{
    background-color: {secondary};
    color: {bg};
    font-weight: bold;
}}
QPushButton#stopBtn {{
    background-color: {accent};
    color: {bg};
    font-weight: bold;
}}
QPushButton#trigBtn {{
    background-color: #f9e2af;
    color: {bg};
    font-weight: bold;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
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
    """Primary application window."""

    def __init__(self, intake_payload=None):
        super().__init__()
        self.setWindowTitle("USB Serial ECG Dashboard")
        self.setMinimumSize(1200, 800)

        self._intake_payload = intake_payload or {}

        self._frozen    = False
        self._connected = False
        self._window_seconds = DEFAULT_WINDOW_SECONDS

        self._last_hrv_result: dict = {}
        self._last_window_result: dict = {}

        # Serial HR samples accumulated between 5-second exports
        self._hr_5s_buf: list = []

        self._exporter = DataExporter()

        # Ring buffers for plotting (120 s)
        buf_sec = 120
        self._ecg_buf   = RingBuffer(ECG_NATIVE_HZ * buf_sec)
        self._acc_x_buf = RingBuffer(ACC_HZ * buf_sec)
        self._acc_y_buf = RingBuffer(ACC_HZ * buf_sec)
        self._acc_z_buf = RingBuffer(ACC_HZ * buf_sec)
        self._hr_buf    = RingBuffer(buf_sec)

        # Workers
        self._serial_worker = None
        self._serial_thread = None
        self._proc_worker   = None
        self._proc_thread   = None
        self._mqtt_worker   = None

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

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._build_toolbar(plot_layout)
        self._build_plots(plot_layout)
        splitter.addWidget(plot_widget)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._build_device_controls(right_layout)
        self._build_recording_panel(right_layout)
        self._build_intake_summary_panel(right_layout)
        self._build_hrv_panel(right_layout)
        self._build_log_panel(right_layout)
        right_layout.addStretch()
        splitter.addWidget(right_panel)

        splitter.setSizes([900, 350])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Select a serial port or use mock mode to begin.")

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

        self._ecg_plot = self._plot_widget.addPlot(row=0, col=0, title="ECG (100 Hz, 0.5-25 Hz BP)")
        self._ecg_plot.setLabel("left", "Amplitude")
        self._ecg_plot.showGrid(x=True, y=True, alpha=0.15)
        self._ecg_plot.setDownsampling(auto=True, mode="peak")
        self._ecg_plot.setClipToView(True)
        self._ecg_curve = self._ecg_plot.plot(
            pen=pg.mkPen(DARK_THEME["ecg_color"], width=1.5)
        )

        self._acc_plot = self._plot_widget.addPlot(row=1, col=0, title="Accelerometer (100 Hz)")
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
        group = QGroupBox("Serial Connection")
        layout = QVBoxLayout()

        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan Ports")
        self._scan_btn.clicked.connect(self._on_scan)
        scan_row.addWidget(self._scan_btn)

        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(180)
        self._device_combo.addItem("No ports found")
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

        # Trigger button
        trig_row = QHBoxLayout()
        self._trigger_btn = QPushButton("Send TRIG")
        self._trigger_btn.setObjectName("trigBtn")
        self._trigger_btn.setToolTip(
            f"Send TRIG command to ESP32 to fire BIOPAC trigger pulse on A0"
        )
        self._trigger_btn.clicked.connect(self._on_trigger)
        self._trigger_btn.setEnabled(False)
        trig_row.addWidget(self._trigger_btn)

        baud_label = QLabel(f"Baud: {SERIAL_BAUD_RATE}")
        baud_label.setStyleSheet(f"color: {DARK_THEME['text_dim']}; font-size: 11px;")
        trig_row.addWidget(baud_label)
        trig_row.addStretch()
        layout.addLayout(trig_row)

        self._conn_label = QLabel("Status: Disconnected")
        self._conn_label.setStyleSheet(f"color: {DARK_THEME['accent']};")
        layout.addWidget(self._conn_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_recording_panel(self, parent_layout):
        group = QGroupBox("Recording")
        layout = QVBoxLayout()
        layout.setSpacing(6)

        rec_row = QHBoxLayout()
        self._record_btn = QPushButton("  Start Recording")
        self._record_btn.setObjectName("recordBtn")
        self._record_btn.clicked.connect(self._on_start_recording)
        rec_row.addWidget(self._record_btn)

        self._stop_rec_btn = QPushButton("  Stop")
        self._stop_rec_btn.setObjectName("stopBtn")
        self._stop_rec_btn.clicked.connect(self._on_stop_recording)
        self._stop_rec_btn.setEnabled(False)
        rec_row.addWidget(self._stop_rec_btn)
        layout.addLayout(rec_row)

        self._rec_status_lbl = QLabel("Not recording")
        self._rec_status_lbl.setStyleSheet(f"color: {DARK_THEME['text_dim']}; font-size: 11px;")
        self._rec_status_lbl.setWordWrap(True)
        layout.addWidget(self._rec_status_lbl)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_intake_summary_panel(self, parent_layout):
        group = QGroupBox("Patient Profile")
        layout = QVBoxLayout()
        
        sub_id = self._intake_payload.get("subject_id", "Unknown")
        age = self._intake_payload.get("age", "--")
        sex = self._intake_payload.get("sex", "--")
        
        self._profile_info_lbl = QLabel(f"Subject ID: {sub_id}\nAge: {age} | Sex: {sex}")
        self._profile_info_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']}; font-weight: bold; font-size: 13px;")
        layout.addWidget(self._profile_info_lbl)
        
        btn_layout = QHBoxLayout()
        self._edit_intake_btn = QPushButton("Edit Intake Form")
        self._edit_intake_btn.clicked.connect(self._on_edit_intake)
        btn_layout.addStretch()
        btn_layout.addWidget(self._edit_intake_btn)
        
        layout.addLayout(btn_layout)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _on_edit_intake(self):
        from serial_ecg.ui.intake_form import IntakeFormDialog
        dlg = IntakeFormDialog(self)
        if dlg.exec_() == IntakeFormDialog.Accepted:
            self._intake_payload = dlg.payload
            sub_id = self._intake_payload.get("subject_id", "Unknown")
            age = self._intake_payload.get("age", "--")
            sex = self._intake_payload.get("sex", "--")
            self._profile_info_lbl.setText(f"Subject ID: {sub_id}\nAge: {age} | Sex: {sex}")

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

        # Tab 1: HRV metrics
        hrv_tab  = QWidget()
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
        for i, (label, key, _) in enumerate(hrv_fields):
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

        # Tab 2: ECG morphology
        ecg_tab  = QWidget()
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
        for i, (label, key, _) in enumerate(ecg_fields):
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

        # Tab 3: Signal quality
        sqi_tab  = QWidget()
        sqi_grid = QGridLayout(sqi_tab)
        sqi_grid.setContentsMargins(8, 10, 8, 8)
        sqi_grid.setHorizontalSpacing(16)
        sqi_grid.setVerticalSpacing(2)

        sqi_hdr = QLabel("ECG SQI (NeuroKit)")
        sqi_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        sqi_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(sqi_hdr, 0, 0)

        self._sqi_val_lbl = QLabel("--")
        self._sqi_val_lbl.setFont(QFont("Consolas", 14))
        self._sqi_val_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']};")
        sqi_grid.addWidget(self._sqi_val_lbl, 1, 0)

        sqi_cat_hdr = QLabel("Quality")
        sqi_cat_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        sqi_cat_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(sqi_cat_hdr, 0, 1)

        self._sqi_cat_lbl = QLabel("--")
        self._sqi_cat_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sqi_grid.addWidget(self._sqi_cat_lbl, 1, 1)
        
        qrs_hdr = QLabel("QRS Energy")
        qrs_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        qrs_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(qrs_hdr, 2, 0)
        
        self._qrs_val_lbl = QLabel("--")
        self._qrs_val_lbl.setFont(QFont("Consolas", 11))
        self._qrs_val_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']};")
        sqi_grid.addWidget(self._qrs_val_lbl, 3, 0)
        
        vital_hdr = QLabel("Vital Kurtosis")
        vital_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        vital_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(vital_hdr, 2, 1)
        
        self._vital_val_lbl = QLabel("--")
        self._vital_val_lbl.setFont(QFont("Consolas", 11))
        self._vital_val_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']};")
        sqi_grid.addWidget(self._vital_val_lbl, 3, 1)

        inst_hr_hdr = QLabel("ECG HR (5 s)")
        inst_hr_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        inst_hr_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(inst_hr_hdr, 4, 0)

        self._inst_hr_lbl = QLabel("--")
        self._inst_hr_lbl.setFont(QFont("Consolas", 11))
        self._inst_hr_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']};")
        sqi_grid.addWidget(self._inst_hr_lbl, 5, 0)
        
        act_hdr = QLabel("HAR Activity")
        act_hdr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        act_hdr.setStyleSheet(f"color: {DARK_THEME['text_dim']};")
        sqi_grid.addWidget(act_hdr, 4, 1)

        self._act_lbl = QLabel("--")
        self._act_lbl.setFont(QFont("Consolas", 10))
        self._act_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']};")
        sqi_grid.addWidget(self._act_lbl, 5, 1)

        sqi_grid.setRowStretch(6, 1)
        tabs.addTab(sqi_tab, "Quality")

        outer.addWidget(tabs)
        group.setLayout(outer)
        parent_layout.addWidget(group)

    def _build_log_panel(self, parent_layout):
        group = QGroupBox("Log")
        layout = QVBoxLayout()
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(140)
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
    #  Plot timer
    # ------------------------------------------------------------------ #

    def _start_plot_timer(self):
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(33)

    def _update_plots(self):
        if self._frozen:
            return

        win   = self._window_seconds
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
        self._log("Scanning serial ports...")

        worker = SerialWorker(use_mock=False)
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
    def _on_device_found(self, name, port):
        self._device_combo.addItem(f"{name} ({port})", port)
        self._log(f"Found: {name} [{port}]")

    def _on_connect(self):
        port = self._device_combo.currentData()
        if not port:
            self._log("No port selected")
            return
        self._start_acquisition(use_mock=False, address=port)

    def _on_mock_connect(self):
        self._start_acquisition(use_mock=True)

    def _on_trigger(self):
        """Send trigger command to the ESP32 via serial."""
        if self._serial_worker:
            self._serial_worker.send_trigger()

    def _start_acquisition(self, use_mock: bool, address: str = None):
        if self._serial_thread and self._serial_thread.isRunning():
            self._log("Already connected")
            return

        self._start_mqtt()

        self._serial_worker = SerialWorker(use_mock=use_mock)
        if address:
            self._serial_worker.set_device_address(address)

        self._serial_thread = QThread()
        self._serial_worker.moveToThread(self._serial_thread)
        self._serial_thread.started.connect(self._serial_worker.run)

        self._serial_worker.ecg_data.connect(self._on_ecg_data)
        self._serial_worker.acc_data.connect(self._on_acc_data)
        self._serial_worker.hr_data.connect(self._on_hr_data)
        self._serial_worker.status.connect(self._log)
        self._serial_worker.connected.connect(self._on_connection_changed)

        self._proc_worker = ProcessingWorker()
        self._proc_thread = QThread()
        self._proc_worker.moveToThread(self._proc_thread)
        self._proc_thread.started.connect(self._proc_worker.run)

        self._proc_worker.hrv_result.connect(self._on_hrv_result)
        self._proc_worker.window_result.connect(self._on_window_result)
        self._proc_worker.status.connect(self._log)

        self._proc_thread.start()
        self._serial_thread.start()

        self._connect_btn.setEnabled(False)
        self._mock_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._freeze_btn.setEnabled(True)
        self._trigger_btn.setEnabled(True)

    def _on_disconnect(self):
        if self._serial_worker:
            self._serial_worker.stop()
        if self._proc_worker:
            self._proc_worker.stop()

        if self._serial_thread:
            self._serial_thread.quit()
            self._serial_thread.wait(3000)
        if self._proc_thread:
            self._proc_thread.quit()
            self._proc_thread.wait(3000)

        self._serial_worker = None
        self._proc_worker   = None

        self._connect_btn.setEnabled(True)
        self._mock_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._freeze_btn.setEnabled(False)
        self._trigger_btn.setEnabled(False)

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
        _ts, samples = data
        self._ecg_buf.extend(samples)
        if self._proc_worker:
            self._proc_worker.add_raw_ecg(samples)

    @pyqtSlot(object)
    def _on_acc_data(self, data):
        _ts, samples = data
        if samples:
            arr = np.array(samples, dtype=np.float64)
            self._acc_x_buf.extend(arr[:, 0])
            self._acc_y_buf.extend(arr[:, 1])
            self._acc_z_buf.extend(arr[:, 2])
            if self._proc_worker:
                self._proc_worker.add_raw_acc(arr)

    @pyqtSlot(object)
    def _on_hr_data(self, data):
        _ts, hr, _rr = data
        self._hr_buf.append(hr)
        self._hr_5s_buf.append(float(hr))

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

    # ------------------------------------------------------------------ #
    #  Recording controls
    # ------------------------------------------------------------------ #

    def _start_mqtt(self):
        subject_id = self._intake_payload.get("subject_id", "")
        if not subject_id:
            return
            
        if not getattr(self, '_mqtt_worker', None):
            self._mqtt_worker = MQTTWorker(broker="broker.emqx.io", port=1883)
            self._mqtt_worker.log_msg.connect(self._log)
            self._mqtt_worker.start()
            
        self._mqtt_worker.publish(f"pulseforgeai/{subject_id}/info", self._intake_payload)

    def _on_start_recording(self):
        subject_id = self._intake_payload.get("subject_id", "")
        if not subject_id:
            self._log("Subject ID is missing. Please edit the Intake Form.")
            return
        try:
            path = self._exporter.start_session(subject_id)
            import json
            try:
                with open(path / "intake_state.json", "w") as f:
                    json.dump(self._intake_payload, f, indent=4)
            except Exception as e:
                self._log(f"Failed to copy intake form to session path: {e}")

        except Exception as exc:
            self._log(f"Cannot start recording: {exc}")
            return

        self._record_btn.setEnabled(False)
        self._stop_rec_btn.setEnabled(True)
        self._edit_intake_btn.setEnabled(False)
        self._rec_status_lbl.setStyleSheet(f"color: {DARK_THEME['secondary']}; font-size: 11px;")
        self._rec_status_lbl.setText(f"Recording  ->  {path.name}")
        self._log(f"Recording started: {path}")
        self._status_bar.showMessage(f"Recording subject {subject_id} -> {path.name}")

    def _on_stop_recording(self):
        self._exporter.stop_session()
        
        if getattr(self, '_mqtt_worker', None):
            self._mqtt_worker.stop()
            self._mqtt_worker = None
            self._log("MQTT connection stopped.")

        n = self._exporter.window_count
        self._record_btn.setEnabled(True)
        self._stop_rec_btn.setEnabled(False)
        self._edit_intake_btn.setEnabled(True)
        self._rec_status_lbl.setStyleSheet(f"color: {DARK_THEME['text_dim']}; font-size: 11px;")
        self._rec_status_lbl.setText(f"Stopped. {n} window(s) saved.")
        self._log(f"Recording stopped. {n} x 5-second windows exported.")
        self._status_bar.showMessage("Recording stopped.")

    # ------------------------------------------------------------------ #
    #  Analysis result handlers
    # ------------------------------------------------------------------ #

    @pyqtSlot(object)
    def _on_hrv_result(self, result: dict):
        self._last_hrv_result = result

        _hrv_fmt = {
            "rmssd":   lambda v: f"{v:.1f} ms",
            "sdnn":    lambda v: f"{v:.1f} ms",
            "lf_hf":   lambda v: f"{v:.2f}",
            "mean_hr": lambda v: f"{v:.0f} bpm",
        }
        for key, lbl in self._hrv_labels.items():
            val = result.get(key)
            lbl.setText(_hrv_fmt[key](val) if val is not None else "--")

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

    @pyqtSlot(object)
    def _on_window_result(self, result: dict):
        """Called every 5 seconds from the processing worker."""
        self._last_window_result = result

        sqi = result.get("sqi")
        sqi_metrics = result.get("sqi_metrics", {})
        
        if sqi is not None:
            self._sqi_val_lbl.setText(f"{sqi:.3f}")
            cat, color = _sqi_category(sqi)
            self._sqi_cat_lbl.setText(cat)
            self._sqi_cat_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self._sqi_val_lbl.setText("--")
            self._sqi_cat_lbl.setText("--")
            self._sqi_cat_lbl.setStyleSheet(f"color: {DARK_THEME['text_dim']}; font-weight: bold;")
            
        qrs_sqi = sqi_metrics.get("qrs_energy")
        vital_sqi = sqi_metrics.get("vital_kurtosis")
        
        if qrs_sqi is not None:
            self._qrs_val_lbl.setText(f"{qrs_sqi:.3f}")
        else:
            self._qrs_val_lbl.setText("--")
            
        if vital_sqi is not None:
            self._vital_val_lbl.setText(f"{vital_sqi:.2f}")
        else:
            self._vital_val_lbl.setText("--")

        inst_hr = result.get("instant_hr")
        self._inst_hr_lbl.setText(f"{inst_hr:.1f} bpm" if inst_hr is not None else "--")

        har_act = result.get("har_activity", {})
        act_label = har_act.get("label", "unknown")
        act_title = act_label.replace("_", " ").title() if act_label != "unknown" else "--"
        if act_label != "unknown":
            max_conf = max(list(har_act.get("confidence", {}).values()) or [0])
            if max_conf == 1.0:
                self._act_lbl.setText(f"{act_title} (Heuristic)")
            else:
                self._act_lbl.setText(f"{act_title} (ML: {max_conf:.0%})")
        else:
            self._act_lbl.setText("--")

        self._export_window(result)

    def _export_window(self, window_result: dict):
        """Build and append a 5-second payload to the JSON export file."""
        hrv = self._last_hrv_result

        hr_samples  = list(self._hr_5s_buf)
        self._hr_5s_buf.clear()
        avg_hr_serial = float(np.mean(hr_samples)) if hr_samples else None

        acc_features = window_result.get("acc_features", {})

        payload = self._exporter.build_payload(
            subject_id     = self._intake_payload.get("subject_id", "unknown"),
            unix_timestamp = window_result.get("timestamp", time.time()),
            window_s       = 5,
            ecg_sqi        = window_result.get("sqi"),
            avg_hr_bpm     = avg_hr_serial,
            n_hr_samples   = len(hr_samples),
            avg_hr_ecg_bpm = window_result.get("instant_hr"),
            n_r_peaks      = window_result.get("n_r_peaks", 0),
            rmssd_ms       = hrv.get("rmssd"),
            sdnn_ms        = hrv.get("sdnn"),
            lf_hf          = hrv.get("lf_hf"),
            qrs_ms         = hrv.get("qrs_width"),
            qt_ms          = hrv.get("qt_width"),
            qtc_ms         = hrv.get("qtc_width"),
            st_ms          = hrv.get("st_width"),
            p_ms           = hrv.get("p_width"),
            acc_features   = acc_features,
            har_activity   = window_result.get("har_activity", {}),
        )
        if self._exporter.is_recording:
            self._exporter.append_window(payload)
        
        if getattr(self, '_mqtt_worker', None):
            subject_id = self._intake_payload.get("subject_id", "unknown")
            mqtt_payload = payload.copy()
            mqtt_payload["raw_ecg"] = window_result.get("raw_ecg", [])
            self._mqtt_worker.publish(f"pulseforgeai/{subject_id}/raw", mqtt_payload)

        if self._exporter.is_recording:
            n = self._exporter.window_count
            sqi = window_result.get("sqi")
            sqi_str = f"  SQI={sqi:.3f}" if sqi is not None else ""
            self._rec_status_lbl.setText(
                f"Recording  [{n} windows]{sqi_str}  ->  {self._exporter.session_file.name}"
            )

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        if self._exporter.is_recording:
            self._exporter.stop_session()
        self._on_disconnect()
        event.accept()


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _sqi_category(sqi: float):
    """Return a human-readable quality label and theme colour."""
    T = DARK_THEME
    if sqi >= 0.60:
        return "Good", T["primary"]
    if sqi > 0.30:
        return "Fair", "#fab387"
    return "Poor", T["accent"]
