"""
targeted_embed.py — 只对 Neo4j 里缺失 m2d2_embedding 或 omar_embedding 的歌曲提取向量

用法：
  # Step 1: 只列出缺向量的歌曲（诊断模式，不写任何数据）
  python data/pipeline/extract_missing_embeddings.py --dry-run

  # Step 2: 提取并写入缺向量的歌曲
  python data/pipeline/extract_missing_embeddings.py
"""
import os
import sys
import glob
import logging
import argparse
from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_AUDIO_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\audio"
DEFAULT_METADATA_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\metadata"
MAX_AUDIO_SECONDS = 180  # 只取前 3 分钟防 OOM


# ---------- Neo4j 查询缺向量的歌曲 ----------
def query_missing_embedding_songs() -> list:
    """返回 Neo4j 里缺少 m2d2 或 omar 向量的所有非 MTG 歌曲信息"""
    client = get_neo4j_client()
    records = client.execute_query("""
        MATCH (s:Song)
        WHERE NOT (s.dataset = 'mtg')
          AND (s.m2d2_embedding IS NULL OR s.omar_embedding IS NULL)
        RETURN s.music_id AS music_id, s.title AS title, s.audio_url AS audio_url
        ORDER BY s.title
    """)
    songs = []
    for r in records:
        songs.append({
            "music_id": r["music_id"],
            "title": r["title"],
            "audio_url": r.get("audio_url", ""),
        })
    return songs


def find_audio_file(music_id: str, title: str, audio_dir: str) -> str | None:
    """根据 music_id 或 title 定位本地音频文件"""
    # 策略1: 精确匹配文件名含 music_id 的文件（目前命名规则是 title - artist.mp3）
    # 策略2: 扫描整个 audio_dir 找到元数据里 musicId 匹配的
    meta_dir = DEFAULT_METADATA_DIR
    for meta_file in glob.glob(os.path.join(meta_dir, "*_meta.json")):
        import json
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if str(meta.get("musicId", "")) == str(music_id):
                basename = os.path.basename(meta_file).replace("_meta.json", "")
                for ext in ("mp3", "flac", "wav", "ogg"):
                    candidate = os.path.join(audio_dir, f"{basename}.{ext}")
                    if os.path.exists(candidate):
                        return candidate
        except Exception:
            continue
    return None


def extract_and_write(audio_path: str, music_id: str, title: str, dry_run: bool):
    """提取音频向量并写入 Neo4j"""
    if dry_run:
        logger.info(f"  [DRY-RUN] 会处理: {title} ({os.path.basename(audio_path)})")
        return

    import librosa
    from retrieval.audio_embedder import encode_audio_to_embedding, extract_audio_representation

    try:
        file_duration = librosa.get_duration(path=audio_path)
        load_duration = MAX_AUDIO_SECONDS if file_duration > MAX_AUDIO_SECONDS else None
        audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)
        audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        m2d2_emb = encode_audio_to_embedding(audio_16k, sample_rate=16000)
        omar_emb  = extract_audio_representation(audio_16k, sample_rate=16000)

        # 写回 Neo4j
        client = get_neo4j_client()
        client.execute_query("""
            MATCH (s:Song {music_id: $music_id})
            SET s.m2d2_embedding = $m2d2, s.omar_embedding = $omar, s.updated_at = timestamp()
        """, {"music_id": music_id, "m2d2": m2d2_emb, "omar": omar_emb})

        logger.info(f"  ✅ {title}: M2D2 {len(m2d2_emb)}维, OMAR {len(omar_emb)}维")

    except Exception as e:
        logger.error(f"  ❌ 失败 {title}: {e}")


def main():
    parser = argparse.ArgumentParser(description="对 Neo4j 缺失向量的歌曲补提取 M2D2+OMAR 向量")
    parser.add_argument("--dry-run", action="store_true", help="只列出缺向量的歌曲，不进行任何提取")
    parser.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR)
    args = parser.parse_args()

    logger.info("🔍 查询 Neo4j 中缺少向量的歌曲...")
    missing = query_missing_embedding_songs()

    if not missing:
        logger.info("🎉 所有非 MTG 歌曲均已有向量，无需补提取！")
        return

    logger.info(f"📋 共找到 {len(missing)} 首歌曲缺少向量：")
    for s in missing:
        logger.info(f"   - {s['title']} (music_id={s['music_id']})")

    if args.dry_run:
        logger.info("\n[DRY-RUN 模式] 未进行任何提取，以上即为需要处理的列表。")
        return

    # 加载模型
    logger.info("\n🧠 加载 M2D-CLAP + OMAR-RQ 模型...")
    from retrieval.audio_embedder import get_m2d2_model, get_omar_model
    get_m2d2_model()
    get_omar_model()
    logger.info("✅ 模型加载完毕，开始逐首提取...")

    found = 0
    not_found = 0
    for song in missing:
        audio_path = find_audio_file(song["music_id"], song["title"], args.audio_dir)
        if audio_path:
            found += 1
            extract_and_write(audio_path, song["music_id"], song["title"], args.dry_run)
        else:
            not_found += 1
            logger.warning(f"  ⚠️ 找不到音频文件: {song['title']} (music_id={song['music_id']})")

    logger.info(f"\n🏁 完成！处理 {found} 首，找不到音频 {not_found} 首。")


if __name__ == "__main__":
    main()
