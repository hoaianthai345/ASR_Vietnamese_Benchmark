from vnasrbench.streaming.stitch import compute_stability, lcs_merge_tokens


def test_lcs_merge_tokens_overlap() -> None:
    prev = "a b c".split()
    cur = "b c d".split()
    merged = lcs_merge_tokens(prev, cur, tail_window_tokens=50)
    assert merged == "a b c d".split()


def test_lcs_merge_tokens_no_overlap() -> None:
    prev = "a b".split()
    cur = "c d".split()
    merged = lcs_merge_tokens(prev, cur, tail_window_tokens=50)
    assert merged == "a b c d".split()


def test_compute_stability() -> None:
    st = compute_stability(["a", "a b", "a b"])
    assert st.edit_overhead == 0.5
    assert st.change_rate == 0.5
    assert st.avg_step_edits == 0.5

