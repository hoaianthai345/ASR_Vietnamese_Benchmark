from vnasrbench.eval.wer import wer, cer


def test_wer_zero() -> None:
    r = ["xin chào"]
    h = ["xin chào"]
    assert wer(r, h).wer == 0.0


def test_cer_zero() -> None:
    r = ["xin chào"]
    h = ["xin chào"]
    assert cer(r, h).cer == 0.0


def test_cer_simple() -> None:
    r = ["ab"]
    h = ["ac"]
    # one substitution over 2 chars -> 0.5
    assert cer(r, h).cer == 0.5
