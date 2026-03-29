import json
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QWidget, QFormLayout, QSpinBox, QComboBox, QDoubleSpinBox,
    QDateEdit, QCheckBox, QMessageBox, QGroupBox, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QDate

from serial_ecg.utils.constants import DARK_THEME

INTAKE_JSON_PATH = Path("intake_state.json")

class IntakeFormDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Patient Intake Form")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(self._build_stylesheet())
        
        self.payload = {}
        
        self._build_ui()
        self._load_from_json()

    def _build_stylesheet(self):
        return f"""
        QDialog {{ background-color: {DARK_THEME['background']}; color: {DARK_THEME['text']}; }}
        QLabel {{ color: {DARK_THEME['text']}; font-family: 'Noto Sans'; font-size: 13px; }}
        QGroupBox {{ border: 1px solid {DARK_THEME['border']}; border-radius: 6px; margin-top: 10px; color: {DARK_THEME['text']}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{ 
            background-color: {DARK_THEME['surface']}; color: {DARK_THEME['text']}; 
            border: 1px solid {DARK_THEME['border']}; border-radius: 4px; padding: 6px; 
            font-family: 'Noto Sans'; font-size: 13px;
        }}
        QComboBox QAbstractItemView {{ background-color: {DARK_THEME['surface']}; color: {DARK_THEME['text']}; }}
        QPushButton {{ background-color: {DARK_THEME['surface']}; color: {DARK_THEME['text']}; border: 1px solid {DARK_THEME['border']}; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-family: 'Noto Sans'; font-size: 13px; }}
        QPushButton:hover {{ background-color: {DARK_THEME['border']}; }}
        QPushButton#clearBtn {{ background-color: #EF4444; color: white; border: none; }}
        QPushButton#clearBtn:hover {{ background-color: #DC2626; }}
        QPushButton#saveBtn {{ background-color: {DARK_THEME['primary']}; color: {DARK_THEME['background']}; border: none; }}
        QPushButton#saveBtn:hover {{ background-color: #2563EB; }}
        QTabWidget::pane {{ border: 1px solid {DARK_THEME['border']}; background: {DARK_THEME['background']}; border-radius: 4px; }}
        QTabBar::tab {{ background: {DARK_THEME['surface']}; color: {DARK_THEME['text_dim']}; padding: 8px 16px; border: 1px solid {DARK_THEME['border']}; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; font-family: 'Noto Sans'; }}
        QTabBar::tab:selected {{ background: {DARK_THEME['background']}; color: {DARK_THEME['text']}; font-weight: bold; }}
        QCheckBox {{ color: {DARK_THEME['text']}; font-family: 'Noto Sans'; }}
        """

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        
        id_layout = QHBoxLayout()
        id_label = QLabel("Subject ID:")
        id_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.subject_id_edit = QLineEdit()
        self.subject_id_edit.setPlaceholderText("REQUIRED - e.g. S001")
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.subject_id_edit)
        main_layout.addLayout(id_layout)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        
        btn_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load JSON...")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.clicked.connect(self._on_load_json)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self._clear_form)
        
        self.save_btn = QPushButton("Save & Continue")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self._on_save)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

    def _build_tab1(self):
        w = QWidget()
        layout = QFormLayout(w)
        layout.setSpacing(12)
        
        self.f_age = QSpinBox(); self.f_age.setRange(1, 120); self.f_age.setValue(50)
        self.f_sex = QComboBox(); self.f_sex.addItems(["Male", "Female", "Other"])
        self.f_height = QSpinBox(); self.f_height.setRange(50, 250); self.f_height.setValue(170); self.f_height.setSuffix(" cm")
        self.f_weight = QDoubleSpinBox(); self.f_weight.setRange(20.0, 300.0); self.f_weight.setValue(70.0); self.f_weight.setSuffix(" kg")
        
        hr_layout = QHBoxLayout()
        self.f_hr_low = QSpinBox(); self.f_hr_low.setRange(30, 200); self.f_hr_low.setValue(80)
        self.f_hr_high = QSpinBox(); self.f_hr_high.setRange(30, 200); self.f_hr_high.setValue(130)
        hr_layout.addWidget(QLabel("Low:"))
        hr_layout.addWidget(self.f_hr_low)
        hr_layout.addWidget(QLabel("High:"))
        hr_layout.addWidget(self.f_hr_high)
        
        layout.addRow("Q1: Patient Age:", self.f_age)
        layout.addRow("Q2: Biological Sex:", self.f_sex)
        layout.addRow("Q3: Height (cm):", self.f_height)
        layout.addRow("Q4: Current Weight (kg):", self.f_weight)
        layout.addRow("Q5: Target Heart Rate Range:", hr_layout)
        
        self.tabs.addTab(w, "Demographics & Physicals")

    def _build_tab2(self):
        w = QWidget()
        layout = QFormLayout(w)
        layout.setSpacing(12)
        
        self.f_event = QComboBox()
        self.f_event.addItems(["Post-MI", "Coronary Bypass", "Stent", "Stable Angina", "Heart Failure", "None"])
        
        self.f_date = QDateEdit()
        self.f_date.setCalendarPopup(True)
        self.f_date.setDate(QDate.currentDate())
        
        self.f_lvef = QSpinBox(); self.f_lvef.setRange(0, 100); self.f_lvef.setValue(55); self.f_lvef.setSuffix(" %")
        
        self.f_comorb_dia = QCheckBox("Diabetes")
        self.f_comorb_copd = QCheckBox("COPD")
        self.f_comorb_hyp = QCheckBox("Hypertension")
        self.f_comorb_pad = QCheckBox("PAD")
        self.f_comorb_ren = QCheckBox("Renal Disease")
        comorb_layout = QVBoxLayout()
        for cb in [self.f_comorb_dia, self.f_comorb_copd, self.f_comorb_hyp, self.f_comorb_pad, self.f_comorb_ren]:
            comorb_layout.addWidget(cb)
            
        self.f_beta = QComboBox(); self.f_beta.addItems(["No", "Yes"])
        
        layout.addRow("Q6: Qualifying Event:", self.f_event)
        layout.addRow("Q7: Date of Event:", self.f_date)
        layout.addRow("Q8: LVEF (%):", self.f_lvef)
        layout.addRow("Q9: Co-morbidities:", comorb_layout)
        layout.addRow("Q10: Current Beta-Blockers:", self.f_beta)
        
        self.tabs.addTab(w, "Clinical History")

    def _build_tab3(self):
        w = QWidget()
        layout = QFormLayout(w)
        layout.setSpacing(12)
        
        self.f_tobacco = QComboBox(); self.f_tobacco.addItems(["Never", "Former", "Current"])
        self.f_activity = QSpinBox(); self.f_activity.setRange(1, 5); self.f_activity.setValue(3)
        self.f_activity.setToolTip("1: Sedentary, 5: Highly Active")
        self.f_chest = QComboBox(); self.f_chest.addItems(["None", "With Exertion", "At Rest"])
        self.f_dyspnea = QComboBox(); self.f_dyspnea.addItems(["None", "On Exertion", "Orthopnea"])
        self.f_phq2 = QSpinBox(); self.f_phq2.setRange(0, 6); self.f_phq2.setValue(0)
        
        layout.addRow("Q11: Tobacco History:", self.f_tobacco)
        layout.addRow("Q12: Baseline Activity (1-5):", self.f_activity)
        layout.addRow("Q13: Chest Pain/Angina (7 Days):", self.f_chest)
        layout.addRow("Q14: Dyspnea (Shortness of Breath):", self.f_dyspnea)
        layout.addRow("Q15: Mood/Interest Screen (PHQ-2):", self.f_phq2)
        
        self.tabs.addTab(w, "Risk & Symptoms")

    def _build_tab4(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        
        instructions = QLabel(
            "<b>Google Fit Historical Sync</b><br><br>"
            "1. Go to <a href='https://console.cloud.google.com/'>Google Cloud Console</a>.<br>"
            "2. Enable 'Fitness API' and setup OAuth Consent Screen.<br>"
            "3. Create OAuth Desktop App credentials and download <code>client_secret.json</code>.<br>"
            "4. Select it below to sync historical steps, sleep, and heart data."
        )
        instructions.setOpenExternalLinks(True)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        self.f_timeframe = QComboBox()
        self.f_timeframe.addItems(["7_days", "1_month"])
        
        tf_layout = QHBoxLayout()
        tf_layout.addWidget(QLabel("Timeframe:"))
        tf_layout.addWidget(self.f_timeframe)
        layout.addLayout(tf_layout)
        
        btn_layout = QHBoxLayout()
        self.btn_select_secret = QPushButton("1. Select client_secret.json")
        self.btn_select_secret.clicked.connect(self._select_client_secret)
        
        self.btn_sync_fit = QPushButton("2. Connect & Sync")
        self.btn_sync_fit.clicked.connect(self._sync_google_fit)
        
        btn_layout.addWidget(self.btn_select_secret)
        btn_layout.addWidget(self.btn_sync_fit)
        layout.addLayout(btn_layout)
        
        self.lbl_secret_path = QLabel("No secret loaded.")
        self.lbl_secret_path.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.lbl_secret_path)
        
        self.txt_fit_status = QTextEdit()
        self.txt_fit_status.setReadOnly(True)
        layout.addWidget(self.txt_fit_status)
        
        self.tabs.addTab(w, "Historical Baseline (Fit)")
        
        self._client_secret_path = "client_secret.json"
        
    def _select_client_secret(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select client_secret.json", "", "JSON Files (*.json)")
        if path:
            self._client_secret_path = path
            self.lbl_secret_path.setText(f"Loaded: {os.path.basename(path)}")

    def _sync_google_fit(self):
        self.txt_fit_status.append("Initializing OAuth flow (check web browser)...")
        self.repaint()
        
        class UIStdOutLogger:
            def __init__(self, callback):
                self.callback = callback
            def write(self, s):
                if s.strip():
                    self.callback(s)
            def flush(self):
                pass
                
        def on_print(msg):
            self.txt_fit_status.append(msg.strip())
            self.repaint()
        
        old_stdout = sys.stdout
        sys.stdout = UIStdOutLogger(on_print)
        
        try:
            from serial_ecg.utils.google_fit_fetcher import GoogleFitFetcher
            fetcher = GoogleFitFetcher(client_secret_path=self._client_secret_path)
            
            if fetcher.authenticate():
                self.txt_fit_status.append("\nAuthenticated successfully. Fetching data...")
                self.repaint()
                
                tf = self.f_timeframe.currentText()
                data = fetcher.fetch_historical_summary(timeframe=tf)
                
                self._historical_data = data
                
                total_steps = sum(d["steps"] for d in data["days"])
                overall_sleep = sum(d["sleep_hours"] for d in data["days"])
                hr_days = sum(1 for d in data["days"] if d.get("avg_bpm") is not None)
                temp_days = sum(1 for d in data["days"] if d.get("body_temp") is not None)
                
                hr_pts = sum(sum(1 for v in d.get("hr_array", {}).get("values", []) if v is not None) for d in data["days"])
                temp_pts = sum(sum(1 for v in d.get("temp_array", {}).get("values", []) if v is not None) for d in data["days"])
                
                deep_hrs = sum(d["sleep_stages"]["deep"] for d in data["days"])
                rem_hrs = sum(d["sleep_stages"]["rem"] for d in data["days"])
                light_hrs = sum(d["sleep_stages"]["light"] for d in data["days"])
                
                self.txt_fit_status.append(f"Success! Fetched {len(data['days'])} days of metrics.")
                self.txt_fit_status.append(f"Total Steps: {total_steps:,}")
                self.txt_fit_status.append(f"Total Sleep: {overall_sleep:.1f} hrs")
                self.txt_fit_status.append(f"Sleep Stages: Deep {deep_hrs:.1f}h | REM {rem_hrs:.1f}h | Light {light_hrs:.1f}h")
                self.txt_fit_status.append(f"HR Array: {hr_pts:,} pts (across {hr_days} days)")
                self.txt_fit_status.append(f"Body Temp Array: {temp_pts:,} pts (across {temp_days} days)")
                
        except Exception as e:
            self.txt_fit_status.append(f"Error: {e}")
        finally:
            sys.stdout = old_stdout

    def _clear_form(self):
        self.subject_id_edit.clear()
        self.f_age.setValue(50)
        self.f_sex.setCurrentIndex(0)
        self.f_height.setValue(170)
        self.f_weight.setValue(70.0)
        self.f_hr_low.setValue(80)
        self.f_hr_high.setValue(130)
        
        self.f_event.setCurrentIndex(0)
        self.f_date.setDate(QDate.currentDate())
        self.f_lvef.setValue(55)
        self.f_comorb_dia.setChecked(False)
        self.f_comorb_copd.setChecked(False)
        self.f_comorb_hyp.setChecked(False)
        self.f_comorb_pad.setChecked(False)
        self.f_comorb_ren.setChecked(False)
        self.f_beta.setCurrentIndex(0)
        
        self.f_tobacco.setCurrentIndex(0)
        self.f_activity.setValue(3)
        self.f_chest.setCurrentIndex(0)
        self.f_dyspnea.setCurrentIndex(0)
        self.f_phq2.setValue(0)

    def _pack_state(self):
        payload = {
            "subject_id": self.subject_id_edit.text().strip(),
            "age": self.f_age.value(),
            "sex": self.f_sex.currentText(),
            "height_cm": self.f_height.value(),
            "weight_kg": self.f_weight.value(),
            "hr_target_low": self.f_hr_low.value(),
            "hr_target_high": self.f_hr_high.value(),
            
            "event": self.f_event.currentText(),
            "event_date": self.f_date.date().toString(Qt.ISODate),
            "lvef": self.f_lvef.value(),
            "comorb_dia": self.f_comorb_dia.isChecked(),
            "comorb_copd": self.f_comorb_copd.isChecked(),
            "comorb_hyp": self.f_comorb_hyp.isChecked(),
            "comorb_pad": self.f_comorb_pad.isChecked(),
            "comorb_ren": self.f_comorb_ren.isChecked(),
            "beta_blocker": self.f_beta.currentText(),
            
            "tobacco": self.f_tobacco.currentText(),
            "activity_level": self.f_activity.value(),
            "chest_pain": self.f_chest.currentText(),
            "dyspnea": self.f_dyspnea.currentText(),
            "phq2": self.f_phq2.value()
        }
        
        if hasattr(self, '_historical_data') and self._historical_data:
            payload["historical_baseline"] = self._historical_data
            
        return payload

    def _on_save(self):
        if not self.subject_id_edit.text().strip():
            QMessageBox.warning(self, "Validation Error", "Subject ID is required.")
            return
            
        self.payload = self._pack_state()
        try:
            with open(INTAKE_JSON_PATH, "w") as f:
                json.dump(self.payload, f, indent=2)
        except Exception as e:
            print(f"Failed to save intake state: {e}")
            
        self.accept()

    def _on_load_json(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Load Intake State", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self._populate_from_dict(data)
                QMessageBox.information(self, "Success", "Participant info loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load JSON:\n{e}")

    def _load_from_json(self):
        if not INTAKE_JSON_PATH.exists():
            return
        
        try:
            with open(INTAKE_JSON_PATH, "r") as f:
                data = json.load(f)
            self._populate_from_dict(data)
        except Exception as e:
            print(f"Failed to load intake state: {e}")

    def _populate_from_dict(self, data):
        """Populates form fields from a provided dictionary."""
        if "subject_id" in data: self.subject_id_edit.setText(data["subject_id"])
        if "age" in data: self.f_age.setValue(data["age"])
        if "sex" in data: self.f_sex.setCurrentText(data["sex"])
        if "height_cm" in data: self.f_height.setValue(data["height_cm"])
        if "weight_kg" in data: self.f_weight.setValue(data["weight_kg"])
        if "hr_target_low" in data: self.f_hr_low.setValue(data["hr_target_low"])
        if "hr_target_high" in data: self.f_hr_high.setValue(data["hr_target_high"])
        
        if "event" in data: self.f_event.setCurrentText(data["event"])
        if "event_date" in data: self.f_date.setDate(QDate.fromString(data["event_date"], Qt.ISODate))
        if "lvef" in data: self.f_lvef.setValue(data["lvef"])
        
        if "comorb_dia" in data: self.f_comorb_dia.setChecked(data["comorb_dia"])
        if "comorb_copd" in data: self.f_comorb_copd.setChecked(data["comorb_copd"])
        if "comorb_hyp" in data: self.f_comorb_hyp.setChecked(data["comorb_hyp"])
        if "comorb_pad" in data: self.f_comorb_pad.setChecked(data["comorb_pad"])
        if "comorb_ren" in data: self.f_comorb_ren.setChecked(data["comorb_ren"])
        if "beta_blocker" in data: self.f_beta.setCurrentText(data["beta_blocker"])
        
        if "tobacco" in data: self.f_tobacco.setCurrentText(data["tobacco"])
        if "activity_level" in data: self.f_activity.setValue(data["activity_level"])
        if "chest_pain" in data: self.f_chest.setCurrentText(data["chest_pain"])
        if "dyspnea" in data: self.f_dyspnea.setCurrentText(data["dyspnea"])
        if "phq2" in data: self.f_phq2.setValue(data["phq2"])
        
        if "historical_baseline" in data: 
            self._historical_data = data["historical_baseline"]
