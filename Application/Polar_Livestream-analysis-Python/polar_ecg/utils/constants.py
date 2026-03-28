"""Application-wide constants for the Polar ECG Dashboard."""

ECG_NATIVE_HZ = 130
ACC_HZ = 100
HR_HZ = 1

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
