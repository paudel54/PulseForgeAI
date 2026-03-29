"""Application-wide constants for the USB Serial ECG Dashboard."""

ECG_NATIVE_HZ = 100
ACC_HZ = 100
HR_HZ = 1

SERIAL_BAUD_RATE = 256000

# CSV column indices (0-based) matching ESP32 firmware header:
# seq,tx_ms,rx_ms,latency_ms,inter_arrival_ms,ECG,Resp0,Resp1,SpO2,HR,
# IR,Red,Temp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ
COL_SEQ   = 0
COL_TX_MS = 1
COL_RX_MS = 2
COL_ECG   = 5
COL_HR    = 9
COL_ACCX  = 13
COL_ACCY  = 14
COL_ACCZ  = 15
MIN_CSV_FIELDS = 16

WINDOW_SECONDS_OPTIONS = [5, 10, 15, 30]
DEFAULT_WINDOW_SECONDS = 10

HRV_ANALYSIS_INTERVAL_S = 2.0

DARK_THEME = {
    "background": "#1e1e2e",
    "surface": "#2a2a3c",
    "primary": "#89b4fa",
    "secondary": "#a6e3a1",
    "accent": "#f38ba8",
    "text": "#cdd6f4",
    "text_dim": "#6c7086",
    "border": "#45475a",
    "plot_bg": "#181825",
    "ecg_color": "#89b4fa",
    "acc_x_color": "#f38ba8",
    "acc_y_color": "#a6e3a1",
    "acc_z_color": "#fab387",
    "hr_color": "#f5c2e7",
    "grid_color": "#313244",
}
