from vnasrbench.utils.textnorm import normalize_vi


def test_normalize_vi_basic() -> None:
    assert normalize_vi("Xin chào!!!") == "xin chào"
    assert normalize_vi("  A  B   ") == "a b"
