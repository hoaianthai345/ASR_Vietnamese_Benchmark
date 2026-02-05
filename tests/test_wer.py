from vnasrbench.eval.wer import wer


def test_wer_zero() -> None:
    r = ["xin chào"]
    h = ["xin chào"]
    assert wer(r, h).wer == 0.0
