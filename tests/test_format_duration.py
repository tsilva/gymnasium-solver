from utils.formatting import format_duration


def test_format_duration_sub_minute():
    assert format_duration(8.98).endswith("seconds")
    assert "8.98" in format_duration(8.98)


def test_format_duration_minute_and_seconds():
    assert format_duration(65) == "1 minute 5 seconds"
    assert format_duration(119) == "1 minute 59 seconds"


def test_format_duration_hours_minutes_seconds():
    # 1 hour, 2 minutes, 25 seconds
    assert format_duration(1 * 3600 + 2 * 60 + 25) == "1 hour 2 minutes 25 seconds"


def test_format_duration_days_hours_minutes_seconds():
    # 1 day, 1 hour, 1 minute, 1 second
    assert format_duration(1 * 86400 + 1 * 3600 + 1 * 60 + 1) == "1 day 1 hour 1 minute 1 second"

