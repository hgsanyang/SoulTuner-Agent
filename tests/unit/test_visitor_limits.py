from api.visitor_limits import VisitorLimits


def test_fixed_window_and_bounded_state():
    now = [0]
    limits = VisitorLimits(clock=lambda: now[0])
    assert limits.allow("a", 1)
    assert not limits.allow("a", 1)
    assert limits.allow("b", 1)
    now[0] = 60
    assert limits.allow("a", 1)
    for i in range(4095):
        assert limits.allow(str(i), 1)
    assert not limits.allow("overflow", 1)


def test_concurrency_is_per_visitor_and_global_and_released():
    limits = VisitorLimits()
    assert limits.enter("a")
    assert not limits.enter("a")
    for name in ("b", "c", "d"):
        assert limits.enter(name)
    assert not limits.enter("e")
    limits.leave("a")
    limits.leave("a")
    assert limits.enter("e")
    assert limits.total_active == 4
