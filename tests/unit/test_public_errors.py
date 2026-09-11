from api.public_errors import public_error


def test_public_error_has_reference_without_exception_details(caplog):
    message = public_error(RuntimeError("private-password-and-path"))
    assert "private-password-and-path" not in message
    assert "错误编号" in message
    assert message != public_error(RuntimeError("private-password-and-path"))
    assert "reference=" in caplog.text
