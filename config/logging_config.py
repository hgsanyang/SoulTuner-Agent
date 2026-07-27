import hashlib
import logging
import os
import sys


def safe_query(query: str) -> str:
    """Redact a user query for logs unless raw logging is explicitly opted in.

    A music request is personal — "我今天特别难过，想一个人静静" is exactly the kind
    of text that should not sit in plaintext logs by default. We log a short hash
    plus length so a line can still be correlated, and only print the raw text
    when MUSIC_LOG_RAW_QUERY is set (dev/debug).
    """
    text = str(query or "")
    if os.getenv("MUSIC_LOG_RAW_QUERY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return text
    if not text:
        return "<empty>"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"<redacted q#{digest} len={len(text)}>"


def _raw_logging_enabled() -> bool:
    return os.getenv("MUSIC_LOG_RAW_QUERY", "0").strip().lower() in {"1", "true", "yes", "on"}


def safe_labels(values) -> str:
    """Redact a user-derived label set (preferred genres, moods, saved tags).

    A preference array is a profile: "喜欢的流派 + 情绪 + 场景" identifies a person
    about as well as the query text does, so it gets the same treatment as
    `safe_query` — count plus a short stable hash, so two log lines can still be
    compared for "same set or not" without printing anyone's taste.
    """
    if isinstance(values, str):
        items = [values] if values.strip() else []
    elif values is None:
        items = []
    else:
        items = [str(v) for v in values if str(v).strip()]
    if _raw_logging_enabled():
        return str(items)
    if not items:
        return "<none>"
    digest = hashlib.sha256("|".join(sorted(items)).encode("utf-8")).hexdigest()[:8]
    return f"<{len(items)} labels #{digest}>"


def safe_filters(**filters) -> str:
    """Report WHICH hard filters are active, never their values.

    `artist_filter='周杰伦'` is user-supplied request content. Which filter slots
    were populated is the useful debugging signal; the values are not.
    """
    if _raw_logging_enabled():
        return ", ".join(f"{k}={v!r}" for k, v in filters.items() if str(v or "").strip()) or "<none>"
    active = [name for name, value in filters.items() if str(value or "").strip()]
    return "+".join(active) if active else "<none>"


def setup_logging():
    """
    配置全局日志记录器。

    该函数为应用程序设置了一个标准化的日志系统。
    - 日志级别设置为 INFO，意味着 INFO, WARNING, ERROR, CRITICAL 级别的日志都将被记录。
    - 日志格式包含时间戳、日志记录器名称、日志级别和消息本身，便于追踪。
    - 日志直接输出到控制台 (stdout)。
    """
    # 创建一个格式化器，定义日志的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建一个处理器，用于将日志记录发送到标准输出（控制台）
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # 获取根日志记录器，并进行配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 防止重复添加处理器
    if not root_logger.handlers:
        root_logger.addHandler(handler)


# 在模块加载时执行一次日志配置
setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    获取一个指定名称的日志记录器实例。

    参数:
        name (str): 通常是当前模块的名称 (__name__)。

    返回:
        logging.Logger: 配置好的日志记录器实例。
    """
    return logging.getLogger(name)
