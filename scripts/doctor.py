#!/usr/bin/env python
"""
环境体检（Preflight Doctor）—— 一眼看清"现在到底缺什么"
========================================================

设计目标：在**任何** Python 环境都能跑（纯标准库，零第三方依赖），
所以哪怕 conda 环境没激活、依赖没装，也能用它定位问题。
特别适合：(1) 你每次启动前自检；(2) 交接给 AI/Codex 时让它先 `python scripts/doctor.py`
看清栈状态，而不是盲目猜测。

用法：
    python scripts/doctor.py

它会检查：conda 环境 / .env 关键项是否填写（不打印值）/ 各服务端口与健康检查 /
模型权重缓存。每个 ❌ 都给一句话修复建议。
"""

import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Windows 控制台常是 GBK 编码，emoji/特殊符号会 UnicodeEncodeError。
# 强制 stdout 用 UTF-8（失败则忽略），保证"任何终端都能跑"。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
# 非 TTY 或 NO_COLOR 时关闭 ANSI 颜色，避免日志里出现裸转义码
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if _USE_COLOR:
    GREEN, RED, YEL, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
else:
    GREEN = RED = YEL = DIM = RST = ""
OK, BAD, WARN = f"{GREEN}✅{RST}", f"{RED}❌{RST}", f"{YEL}⚠️ {RST}"


def _tcp(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http(url: str, timeout: float = 2.0) -> int:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def section(title: str):
    print(f"\n{DIM}{'─'*60}{RST}\n  {title}\n{DIM}{'─'*60}{RST}")


def line(ok: bool, label: str, detail: str = "", fix: str = ""):
    mark = OK if ok else BAD
    print(f"  {mark} {label}" + (f"  {DIM}{detail}{RST}" if detail else ""))
    if not ok and fix:
        print(f"       {YEL}↳ {fix}{RST}")
    return ok


def check_python():
    section("Python / Conda 环境")
    prefix = os.environ.get("CONDA_PREFIX", "")
    env_name = Path(prefix).name if prefix else "(未检测到 conda)"
    is_music = env_name == "music_agent"
    line(is_music, f"conda 环境: {env_name}", f"python={sys.executable}",
         "请先 `conda activate music_agent`（项目依赖装在此环境）")
    line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}",
         fix="建议 Python 3.11（README 指定）")


# .env 解析结果（check_env_file 填充，check_services 复用以探测真实 Neo4j 端口）
ENV: dict = {}


def _read_env(env_path: Path) -> dict:
    out = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def check_env_file():
    section(".env 配置（只看是否填写，绝不打印密钥/口令）")
    env_path = ROOT / ".env"
    if not line(env_path.exists(), ".env 存在", str(env_path),
                "执行 `cp .env.example .env` 并填入 key"):
        return
    ENV.update(_read_env(env_path))
    vals = ENV
    # 至少要有一个 LLM key + Neo4j 口令
    llm_keys = ["SILICONFLOW_API_KEY", "SiliconFlow_API_KEY", "DASHSCOPE_API_KEY",
                "ZHIPU_API_KEY", "GOOGLE_API_KEY", "VOLCENGINE_API_KEY"]
    has_llm = any(vals.get(k) for k in llm_keys)
    line(has_llm, "至少一个 LLM API Key 已填", fix="在 .env 填 SILICONFLOW_API_KEY（或其它厂商）")
    line(bool(vals.get("NEO4J_PASSWORD")), "NEO4J_PASSWORD 已填",
         fix="在 .env 填 NEO4J_PASSWORD（须与 Neo4j 实际口令一致）")
    if vals.get("NEO4J_URI"):
        line(True, f"NEO4J_URI 已填", vals.get("NEO4J_URI"))


def check_services():
    section("服务端口 & 健康检查")
    # Neo4j —— 用 .env 里 NEO4J_URI 的真实端口探测（Neo4j Desktop 常是 4xxxx 随机端口，而非 7687）
    uri = ENV.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    bolt_port = 7687
    try:
        bolt_port = int(uri.rsplit(":", 1)[-1].split("/")[0])
    except Exception:
        pass
    bolt = _tcp("127.0.0.1", bolt_port)
    note = " (Neo4j Desktop 随机端口)" if bolt_port not in (7687,) else ""
    line(bolt, f"Neo4j Bolt :{bolt_port}（数据通道，来自 .env NEO4J_URI{note}）",
         fix="Neo4j 没起来：Mode A→`docker compose up -d neo4j`(→改 .env 为 :7687)；Mode B→打开 Neo4j Desktop 启动数据库")
    if bolt_port != 7687 and _tcp("127.0.0.1", 7687):
        line(False, "⚠️ 端口分裂检测", f".env 指向 :{bolt_port}，但 docker neo4j(:7687) 也在跑",
             "你的 App 连的是 Desktop，docker 那个 neo4j 是另一个库——别搞混。建议二选一")
    browser_code = _http("http://127.0.0.1:7474")
    line(browser_code == 200, "Neo4j Browser :7474（Web UI）", f"HTTP {browser_code or '无响应'}",
         "若 Bolt 通但这里白屏：是浏览器/Desktop UI 问题，非数据库问题（见 doctor 末尾提示）")
    # Backend
    be = _http("http://127.0.0.1:8501/health")
    line(be == 200, "后端 API :8501 /health", f"HTTP {be or '无响应'}",
         "`conda activate music_agent && python start.py --mode api`（或 docker 的 soultuner-backend）")
    # GraphZep（注意：compose=3100，settings 默认=8350，存在端口不一致，两个都探）
    gz3100, gz8350 = _http("http://127.0.0.1:3100/healthcheck"), _http("http://127.0.0.1:8350/healthcheck")
    line(gz3100 == 200 or gz8350 == 200, "GraphZep 记忆服务",
         f":3100→{gz3100 or 'x'} / :8350→{gz8350 or 'x'}",
         "可选服务（挂了不影响核心推荐）。注意 compose 用 3100、settings 默认 8350，建议统一")
    # Frontend
    fe = _tcp("127.0.0.1", 3003) or _tcp("127.0.0.1", 3000)
    line(fe, "前端 :3003/:3000", fix="`cd web && npm run dev`（可选，只跑后端/评测时不需要）")
    # 可选
    line(_tcp("127.0.0.1", 8888), "SearxNG :8888（可选-联网搜索）",
         fix="可选：`docker compose -f docker-compose.searxng.yml up -d`") or None
    line(_tcp("127.0.0.1", 3000), "NeteaseAPI :3000（可选-联网取歌）",
         fix="可选。⚠️注意 :3000 可能与前端 dev 端口冲突") or None


def check_models():
    section("模型权重缓存")
    home = Path.home()
    m2d = home / ".cache" / "m2d_clap"
    hf = home / ".cache" / "huggingface"
    line(m2d.exists() and any(m2d.rglob("*")), "M2D-CLAP 缓存 ~/.cache/m2d_clap",
         fix="`python scripts/download_models.py`（运行时文搜音必需）")
    line(hf.exists(), "HuggingFace 缓存 ~/.cache/huggingface",
         fix="`python scripts/download_models.py`（含 BERT/OMAR；纯查询期非必需）")


def main():
    print(f"\n{'='*60}\n  🩺 SoulTuner-Agent 环境体检\n{'='*60}")
    print(f"  项目根: {ROOT}")
    try:
        check_python()
        check_env_file()
        check_services()
        check_models()
    except Exception as e:
        print(f"\n{RED}体检过程出错: {e}{RST}")
    print(f"\n{DIM}{'─'*60}{RST}")
    print("  💡 Neo4j Browser 白屏排查（Bolt :7687 通但 :7474 白屏 → 数据库没事，是 UI）：")
    print("     1) 硬刷新 Ctrl+Shift+R / 无痕窗口 / 换浏览器（Browser 的 service worker 卡版本最常见）")
    print("     2) 用 cypher-shell 绕过 UI 验证数据：")
    print("        docker exec -it soultuner-neo4j cypher-shell -u neo4j -p <口令> \"MATCH (s:Song) RETURN count(s);\"")
    print("     3) Neo4j Desktop 白屏：重启 Desktop / 清 %APPDATA%\\Neo4j Desktop 缓存 / 看 Desktop 日志")
    print(f"{DIM}{'─'*60}{RST}\n")


if __name__ == "__main__":
    main()
