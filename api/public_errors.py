"""Public error messages never serialize internal exception text."""
import logging
import uuid


def public_error(error: Exception) -> str:
    reference = uuid.uuid4().hex[:12]
    logging.getLogger(__name__).error("Request failed reference=%s", reference, exc_info=error)
    return f"操作暂时无法完成，请稍后重试（错误编号：{reference}）"
