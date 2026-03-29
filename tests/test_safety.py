import json
from backend.safety_engine import EnergySafeWindow

def test_safety_critical_threshold():
    intake = {
        "age": 60,  # HR Max: 160
        "prescribed_intensity_range": [0.4, 0.7] # Safe limit: 112
    }
    engine = EnergySafeWindow(intake)
    
    # Test Exertion > 90% HR Max (160 * 0.9 = 144)
    is_safe, alert, reason = engine.check_safety(hr_bpm=150, activity="exercise", sqi=0.9)
    assert is_safe is False
    assert alert == "critical"

def test_safety_warning_threshold():
    intake = {
        "age": 40,  # HR Max: 180
        "prescribed_intensity_range": [0.5, 0.6] # Safe limit: 108
    }
    engine = EnergySafeWindow(intake)
    
    # Test HR > Prescribed Safe Max (108) but under 90%
    is_safe, alert, reason = engine.check_safety(hr_bpm=115, activity="exercise", sqi=0.9)
    assert is_safe is False
    assert alert == "warning"

def test_quality_advisory_threshold():
    engine = EnergySafeWindow({})
    
    # Test poor SQI
    is_safe, alert, reason = engine.check_safety(hr_bpm=80, activity="rest", sqi=0.4)
    assert is_safe is True
    assert alert == "advisory"

def test_normal_operating_parameters():
    engine = EnergySafeWindow({"age": 50}) # HR Max: 170
    
    is_safe, alert, reason = engine.check_safety(hr_bpm=100, activity="exercise", sqi=0.95)
    assert is_safe is True
    assert alert == "none"
