from posture_logic import (
    calculate_angle,
    classify_posture,
    get_status_class,
    get_status_icon,
)


def test_calculate_angle():
    # Right angle at b (0,0)
    assert abs(calculate_angle((0, 1), (0, 0), (1, 0)) - 90.0) < 1e-9
    # Straight line = 180 deg
    assert abs(calculate_angle((0, 1), (0, 0), (0, -1)) - 180.0) < 1e-9
    # Acute angle ~45 deg at (0,0): a=(0,1), c=(1,1)
    assert abs(calculate_angle((0, 1), (0, 0), (1, 1)) - 45.0) < 1e-9


def test_classify_posture():
    assert classify_posture(160, 0.3) == "Standing"
    assert classify_posture(100, 0.3) == "Sitting"
    assert classify_posture(90, 0.05) == "Sleeping / Lying Down"
    assert classify_posture(140, 0.3) == "Analyzing..."
    assert classify_posture(70, 0.3) == "Analyzing..."
    assert classify_posture(130, 0.3) == "Analyzing..."


def test_status_helpers():
    assert get_status_class("Standing") == "status-standing"
    assert get_status_class("Sitting") == "status-sitting"
    assert get_status_class("Sleeping / Lying Down") == "status-sleeping"
    assert get_status_class("No Person Detected") == "status-unknown"
    assert get_status_icon("Standing") == "🧍"
    assert get_status_icon("Unknown") == "❓"


def test_classify_priority_sleeping():
    # vertical_diff wins even if back_angle suggests standing
    assert classify_posture(160, 0.02) == "Sleeping / Lying Down"


if __name__ == "__main__":
    test_calculate_angle()
    test_classify_posture()
    test_status_helpers()
    test_classify_priority_sleeping()
    print("ALL POSTURE LOGIC TESTS PASSED")