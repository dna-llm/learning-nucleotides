from experiment_one.utils.model_utils import format_param_count


def test_format_param_count():
    assert format_param_count(100_000) == "100K"
    assert format_param_count(900_000) == "900K"
    assert format_param_count(999_999) == "1M"
    assert format_param_count(1_000_000) == "1M"
    assert format_param_count(1_500_000) == "1.5M"
    assert format_param_count(1_000_000_000) == "1B"
    assert format_param_count(1_500_000_000) == "1.5B"
    assert format_param_count(612_000_000) == "612M"
    assert format_param_count(1_234_567_890) == "1.2B"
    assert format_param_count(1_430_000) == "1.4M"
    assert format_param_count(5_610_000) == "5.6M"
    assert format_param_count(22_400_000) == "22M"
    assert format_param_count(14_900_000) == "15M"
    assert format_param_count(84_700_000) == "85M"
