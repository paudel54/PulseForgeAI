"""
Polar ECG Dashboard - POC
Real-time ECG/ACC/HR visualization from a Polar H10 with HRV analysis.

Usage:
    python main.py              # Normal mode (scan for Polar H10)
    python main.py --mock       # Mock sensor mode for testing
"""

import sys
import argparse

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    parser = argparse.ArgumentParser(description="Polar ECG Dashboard")
    parser.add_argument(
        "--mock", action="store_true",
        help="Start with mock sensor (no BLE hardware needed)",
    )
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Polar ECG Dashboard")
    app.setStyle("Fusion")

    from polar_ecg.ui.dashboard import MainDashboard
    from polar_ecg.ui.intake_form import IntakeFormDialog

    dlg = IntakeFormDialog()
    if dlg.exec_() == IntakeFormDialog.Accepted:
        window = MainDashboard(intake_payload=dlg.payload)

        if args.mock:
            window._start_acquisition(use_mock=True)

        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
