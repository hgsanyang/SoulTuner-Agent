"""
SoulTuner-Agent 模型权重一键下载脚本

下载内容：
  1. M2D-CLAP 跨模态模型权重     (~1.6 GB)  — 运行时文本→向量编码
  2. BERT-base-uncased 文本编码器  (~440 MB)  — M2D-CLAP 内部依赖
  3. OMAR-RQ multicodebook 音频模型 (~400 MB)  — 离线数据入库音频特征提取

总计约 2.4 GB，下载完成后 Docker / 本地开发均直接使用，无需重复下载。

注意：GraphZep 嵌入直接调用 SiliconFlow API (BAAI/bge-m3)，无需预下载。

使用方式：
    python scripts/download_models.py
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path


# ---- 配置 ----
M2D_CLAP_URL = (
    "https://github.com/nttcslab/m2d/releases/download/v0.5.0/"
    "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
)
M2D_CLAP_DIR = Path.home() / ".cache" / "m2d_clap"
M2D_CLAP_SUBDIR = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
M2D_CLAP_CHECKPOINT = M2D_CLAP_DIR / M2D_CLAP_SUBDIR / "checkpoint-30.pth"

HF_BERT_MODEL = "google-bert/bert-base-uncased"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"

# HuggingFace 镜像（国内网络加速）
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


OMAR_MODEL_ID = "mtg-upf/omar-rq-multicodebook"


def print_header():
    print("=" * 60)
    print("  SoulTuner-Agent Model Download")
    print("=" * 60)
    print()
    print("  Will download:")
    print("  +------------------------+----------+----------------------+")
    print("  | Model                  | Size     | Purpose              |")
    print("  +------------------------+----------+----------------------+")
    print("  | M2D-CLAP (2025)        | ~1.6 GB  | Text-Audio Encoding  |")
    print("  | BERT-base-uncased      | ~440 MB  | M2D-CLAP Text Enc.   |")
    print("  | OMAR-RQ multicodebook  | ~400 MB  | Audio Feature Embed  |")
    print("  +------------------------+----------+----------------------+")
    print("  | Total                  | ~2.4 GB  |                      |")
    print("  +------------------------+----------+----------------------+")
    print()
    print(f"  M2D-CLAP dir: {M2D_CLAP_DIR}")
    print(f"  HF cache:     {HF_CACHE_DIR}")
    print()


class DownloadProgress:
    """下载进度条"""
    def __init__(self, filename):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        percent = min(int(block_num * block_size * 100 / total_size), 100)
        if percent != self.last_percent:
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            size_mb = block_num * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  [{bar}] {percent}% ({size_mb:.0f}/{total_mb:.0f} MB) {self.filename}"
            )
            sys.stdout.flush()
            self.last_percent = percent
            if percent == 100:
                print()


def download_m2d_clap():
    """下载并解压 M2D-CLAP 模型权重"""
    print("-" * 60)
    print("  [1/3] M2D-CLAP Cross-Modal Model")
    print("-" * 60)

    if M2D_CLAP_CHECKPOINT.exists():
        size_gb = M2D_CLAP_CHECKPOINT.stat().st_size / (1024**3)
        print(f"  [OK] Already exists: {M2D_CLAP_CHECKPOINT} ({size_gb:.1f} GB)")
        print()
        return True

    M2D_CLAP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = M2D_CLAP_DIR / "m2d_clap_2025.zip"

    # 下载 zip
    if not zip_path.exists():
        print(f"  Downloading: {M2D_CLAP_URL}")
        print()
        try:
            urllib.request.urlretrieve(
                M2D_CLAP_URL, str(zip_path), DownloadProgress("m2d_clap.zip")
            )
        except Exception as e:
            print(f"\n  [FAIL] Download failed: {e}")
            print(f"  Please download manually: {M2D_CLAP_URL}")
            print(f"  Extract to: {M2D_CLAP_DIR}/")
            return False
    else:
        print(f"  Zip already exists: {zip_path}")

    # 解压
    print(f"  Extracting...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(M2D_CLAP_DIR))
        print(f"  [OK] Extracted: {M2D_CLAP_DIR / M2D_CLAP_SUBDIR}")
    except Exception as e:
        print(f"  [FAIL] Extraction failed: {e}")
        return False

    print()
    return True


def download_bert():
    """通过 HuggingFace transformers 下载 BERT 文本编码器"""
    print("-" * 60)
    print("  [2/3] BERT-base-uncased Text Encoder")
    print("-" * 60)

    # 检查是否已存在
    bert_cache = HF_CACHE_DIR / "hub" / "models--google-bert--bert-base-uncased"
    if bert_cache.exists():
        print(f"  [OK] Already exists: {bert_cache}")
        print()
        return True

    # 设置 HF 镜像
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"  Downloading {HF_BERT_MODEL} from {HF_MIRROR}...")

    try:
        from transformers import AutoTokenizer, AutoModel

        print("  Downloading tokenizer...")
        AutoTokenizer.from_pretrained(HF_BERT_MODEL)
        print("  Downloading model weights...")
        AutoModel.from_pretrained(HF_BERT_MODEL)
        print(f"  [OK] BERT download complete")
    except ImportError:
        print("  [WARN] transformers not installed, skipping BERT pre-download")
        print("  (Will auto-download on first startup)")
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        print("  (Will auto-download via HF mirror on first startup)")

    print()
    return True


def download_omar_rq():
    """通过 HuggingFace Hub 下载 OMAR-RQ 模型权重"""
    print("-" * 60)
    print("  [3/3] OMAR-RQ Multicodebook Audio Model")
    print("-" * 60)

    # 检查是否已存在
    omar_cache = HF_CACHE_DIR / "hub" / "models--mtg-upf--omar-rq-multicodebook"
    if omar_cache.exists():
        print(f"  [OK] Already exists: {omar_cache}")
        print()
        return True

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"  Downloading {OMAR_MODEL_ID} from {HF_MIRROR}...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(OMAR_MODEL_ID)
        print(f"  [OK] OMAR-RQ download complete")
    except ImportError:
        try:
            # fallback: use omar_rq package directly
            print("  Trying via omar_rq package...")
            from omar_rq import get_model
            import functools, torch
            _orig = torch.load
            torch.load = functools.partial(_orig, weights_only=False)
            try:
                get_model(model_id=OMAR_MODEL_ID, device="cpu")
            finally:
                torch.load = _orig
            print(f"  [OK] OMAR-RQ download complete")
        except Exception as e2:
            print(f"  [WARN] Download failed: {e2}")
            print(f"  Install omar-rq first: pip install omar-rq")
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        print(f"  (Will auto-download on first use during data ingestion)")

    print()
    return True


def print_summary():
    print("=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    print()
    print("  Add these paths to .env (for Docker volume mount):")
    print()
    # 根据操作系统显示不同路径格式
    if sys.platform == "win32":
        m2d_path = str(M2D_CLAP_DIR).replace("\\", "/")
        hf_path = str(HF_CACHE_DIR).replace("\\", "/")
    else:
        m2d_path = str(M2D_CLAP_DIR)
        hf_path = str(HF_CACHE_DIR)
    print(f"    M2D_CLAP_CACHE={m2d_path}")
    print(f"    HF_HOME={hf_path}")
    print()
    print("  Then run: docker compose up -d")
    print()


if __name__ == "__main__":
    print_header()
    
    ok1 = download_m2d_clap()
    ok2 = download_bert()
    ok3 = download_omar_rq()
    
    if ok1 and ok2 and ok3:
        print_summary()
    else:
        print("\n  [WARN] Some downloads failed. Please check network and retry.")
        sys.exit(1)
