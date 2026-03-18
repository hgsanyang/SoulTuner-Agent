"""
一键启动所有服务
=====================
集成启动：前端 / 后端 API / NeteaseAPI / SearxNG / GraphZep

用法：
  python startup_all.py              # 启动全部（GraphZep/SearxNG 可选，不影响核心）
  python startup_all.py --no-docker  # 跳过 Docker 服务（SearxNG）
  python startup_all.py --no-web     # 跳过前端 dev server
  python startup_all.py --no-netease # 跳过 NeteaseAPI（不需要联网音乐时）
"""

import argparse
import os
import subprocess
import sys
import time
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
GRAPHZEP_DIR = PROJECT_ROOT / "graphzep_service"
SEARXNG_COMPOSE = PROJECT_ROOT / "docker-compose.searxng.yml"
# 网易云音乐 API 安装目录
# 优先级：环境变量 NETEASE_API_DIR > 项目根目录下的兼容API服务 > ~/兼容API服务
def _resolve_netease_dir() -> Path:
    env_val = os.environ.get("NETEASE_API_DIR")
    if env_val:
        return Path(env_val)
    local = PROJECT_ROOT / "NeteaseCloudMusicApi"
    if local.exists():
        return local
    # 用户自定义工具目录
    tools_dir = Path(r"C:\Users\sanyang\sanyangworkspace\tools\NeteaseCloudMusicApi")
    if tools_dir.exists():
        return tools_dir
    return Path.home() / "NeteaseCloudMusicApi"

NETEASE_API_DIR = _resolve_netease_dir()

# 子进程列表（用于统一关闭）
_processes = []


def _banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def _start_subprocess(name: str, cmd: list[str], cwd: str | Path, env=None):
    """启动子进程并跟踪"""
    merged_env = {**os.environ, **(env or {})}
    try:
        # Windows: 使用 CREATE_NEW_PROCESS_GROUP 隔离子进程信号，
        # 防止 Uvicorn reloader 重启子进程时信号传播到本脚本导致全部退出
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=merged_env,
            stdout=sys.stdout, stderr=sys.stderr,
            creationflags=creation_flags,
        )
        _processes.append((name, proc))
        print(f"  ✅ {name} 已启动 (PID: {proc.pid})")
        return proc
    except FileNotFoundError:
        print(f"  ⚠️ {name} 启动失败: 命令未找到 ({cmd[0]})")
        return None
    except Exception as e:
        print(f"  ⚠️ {name} 启动失败: {e}")
        return None


def _get_project_python() -> str:
    """
    获取项目专用的 Python 解释器路径。
    优先使用 music_agent conda 环境的 Python（包含 FastAPI 等依赖），
    如果找不到，回退到当前 sys.executable。
    """
    # 优先使用环境变量 CONDA_PREFIX 找 Python，再回退到 sys.executable
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidate = Path(conda_prefix) / "python.exe"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def start_api():
    """启动后端 FastAPI 服务"""
    _banner("启动后端 API 服务 (port 8501)")
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    python_exe = _get_project_python()
    return _start_subprocess(
        "Backend API",
        [python_exe, "start.py", "--mode", "api"],
        PROJECT_ROOT,
    )



def start_web():
    """启动前端 dev server"""
    _banner("启动前端 Web 服务 (port 3000)")
    if not WEB_DIR.exists():
        print(f"  ⚠️ 前端目录不存在: {WEB_DIR}")
        return None
    return _start_subprocess(
        "Frontend Web",
        ["npm.cmd", "run", "dev"],
        WEB_DIR,
    )


def start_searxng():
    """启动 SearxNG Docker 容器"""
    _banner("启动 SearxNG 搜索引擎 (port 8888)")
    if not SEARXNG_COMPOSE.exists():
        print(f"  ⚠️ SearxNG 配置不存在: {SEARXNG_COMPOSE}")
        return None
    return _start_subprocess(
        "SearxNG",
        ["docker-compose", "-f", str(SEARXNG_COMPOSE), "up", "-d"],
        PROJECT_ROOT,
    )


def start_graphzep():
    """启动 GraphZep 记忆服务"""
    _banner("启动 GraphZep 记忆服务 (port 3100)")
    if not GRAPHZEP_DIR.exists():
        print(f"  ⚠️ GraphZep 目录不存在: {GRAPHZEP_DIR}")
        return None
    
    # 检查是否有 docker-compose.yml
    compose_file = GRAPHZEP_DIR / "docker-compose.yml"
    if compose_file.exists():
        return _start_subprocess(
            "GraphZep",
            ["docker-compose", "up", "-d"],
            GRAPHZEP_DIR,
        )
    
    # 检查是否有 npm/node 启动方式
    package_json = GRAPHZEP_DIR / "package.json"
    if package_json.exists():
        return _start_subprocess(
            "GraphZep",
            ["npm.cmd", "start"],
            GRAPHZEP_DIR,
        )
    
    print(f"  ⚠️ GraphZep 无可用启动方式")
    return None


def start_netease_api():
    """启动本地第三方音乐 API 代理服务"""
    _banner("启动 NeteaseAPI 代理服务 (port 3000)")
    if not NETEASE_API_DIR.exists():
        print(f"  ⚠️ NeteaseAPI 目录不存在: {NETEASE_API_DIR}")
        print(f"  💡 请先下载兼容的 API 源码到: {NETEASE_API_DIR}")
        return None
    app_js = NETEASE_API_DIR / "app.js"
    if not app_js.exists():
        print(f"  ⚠️ NeteaseAPI app.js 不存在，请确认安装正确")
        return None
    return _start_subprocess(
        "NeteaseAPI",
        ["npm.cmd", "start"],
        NETEASE_API_DIR,
    )


def cleanup(signum=None, frame=None):
    """统一关闭所有子进程"""
    print("\n\n🛑 正在关闭所有服务...\n")
    for name, proc in reversed(_processes):
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"  ✅ {name} 已关闭")
            except Exception:
                proc.kill()
                print(f"  ⚠️ {name} 强制关闭")
    print("\n全部服务已关闭。")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="一键启动音乐推荐系统所有服务")
    parser.add_argument("--no-docker", action="store_true", help="跳过 Docker 服务（SearxNG）")
    parser.add_argument("--no-web", action="store_true", help="跳过前端 dev server")
    parser.add_argument("--no-graphzep", action="store_true", help="跳过 GraphZep 服务")
    parser.add_argument("--no-netease", action="store_true", help="跳过 NeteaseAPI（不需要联网音乐时）")
    args = parser.parse_args()

    # 注册退出信号
    signal.signal(signal.SIGINT, cleanup)
    # Windows 上 SIGTERM 不可靠，使用 SIGBREAK 代替
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, cleanup)
    else:
        signal.signal(signal.SIGTERM, cleanup)

    _banner("🎵 音乐推荐系统 — 一键启动")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print()

    # 1. NeteaseAPI
    if not args.no_netease:
        start_netease_api()
        time.sleep(2)

    # 2. SearxNG（Docker）
    if not args.no_docker:
        start_searxng()
        time.sleep(1)

    # 3. GraphZep
    if not args.no_graphzep:
        start_graphzep()
        time.sleep(1)

    # 4. 后端 API
    start_api()
    time.sleep(2)

    # 5. 前端
    if not args.no_web:
        start_web()

    _banner("🎉 所有服务已启动！")
    print("  🎵  LocalMusicAPI: http://localhost:3000  (兼容第三方音乐库代理)")
    print("  🖥️  前端:        http://localhost:3003  (Next.js)")
    print("  🔧  API:         http://localhost:8501  (后端)")
    print("  🔍  SearxNG:     http://localhost:8888  (联网搜索)")
    print("  🧠  GraphZep:    http://localhost:3100  (记忆服务)")
    print()
    print("  按 Ctrl+C 关闭所有服务")
    print()

    # 等待所有子进程
    # docker-compose -d 命令启动成功后会立刻以 code=0 退出，这是正常行为
    # 只有 code != 0 且非 docker-compose 才是真正的错误
    _reported_exits = set()
    try:
        while True:
            for name, proc in _processes:
                if proc and proc.poll() is not None and name not in _reported_exits:
                    code = proc.returncode
                    # docker-compose up -d 正常以 0 退出，静默处理
                    if code == 0:
                        pass  # 后台容器正常，无需警告
                    else:
                        print(f"\n  ⚠️ {name} 已退出 (code: {code})")
                    _reported_exits.add(name)
            time.sleep(5)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
