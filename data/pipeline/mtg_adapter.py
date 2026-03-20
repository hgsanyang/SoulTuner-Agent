# ============================================================
# 【MTG-Jamendo 数据集适配器】
#
# 功能：
#   1. 解析 MTG-Jamendo 的 TSV 标注文件
#   2. 随机采样指定数量的歌曲
#   3. 将 MTG 标签映射到系统的 Mood/Theme/Scenario 分类
#   4. 生成与现有 ingest_to_neo4j.py 兼容的 _meta.json
#   5. 输出需要从网盘下载的文件夹列表
#
# 用法：
#   # 采样 3000 首，输出元数据 + 下载清单
#   python data/pipeline/mtg_adapter.py --sample 3000
#
#   # 自定义输出目录
#   python data/pipeline/mtg_adapter.py --sample 3000 --output data/mtg_sample
#
#   # 不采样，处理全部 moodtheme 子集
#   python data/pipeline/mtg_adapter.py --all
# ============================================================

import os
import sys
import shutil
import csv
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MTG_METADATA_DIR = Path(__file__).resolve().parent / "mtg_metadata"

# ---- MTG 标签 → 系统标签映射 ----

# MTG mood/theme 标签到系统 Mood 节点的映射
# 右侧值即 Neo4j Mood 节点名，必须能被 MOOD_TAG_MAP 中的英文关键词通过 CONTAINS 匹配
MTG_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "relaxing": "Relaxing",
    "energetic": "Energetic",
    "melancholic": "Melancholy",
    "calm": "Calm",
    "dark": "Dark",
    "emotional": "Emotional",
    "cool": "Cool",
    "uplifting": "Uplifting",
    "hopeful": "Hopeful",
    "angry": "Angry",
    "aggressive": "Aggressive",
    "tender": "Tender",
    "sexy": "Sexy",
    "funny": "Funny",
    "groovy": "Groovy",
    "powerful": "Powerful",
    "soft": "Soft",
    "positive": "Positive",
    "motivational": "Motivational",
    "inspiring": "Inspiring",
    "sentimental": "Sentimental",
    "eerie": "Eerie",
    "fear": "Fear",
    "holiday": "Holiday",
    "christmas": "Christmas",
    # ── 新补：原来未映射的高频标签 ──
    "romantic": "Romantic",       # 627次，MOOD_TAG_MAP "浪漫"→["romantic","dreamy"] ✅
    "meditative": "Peaceful",     # 742次，映射到 peaceful，MOOD_TAG_MAP "平静"→["peaceful"] ✅
    "fun": "Happy",               # 480次，合并到 Happy
    "upbeat": "Energetic",        # 444次，合并到 Energetic
    "heavy": "Powerful",           # 156次，合并到 Powerful
    "mellow": "Soft",             # 154次，合并到 Soft
    "deep": "Emotional",          # 635次，合并到 Emotional
    "horror": "Eerie",            # 158次，合并到 Eerie
}

# MTG mood/theme 标签到系统 Theme 节点的映射
MTG_TO_THEME = {
    "love": "Love",
    "dream": "Dream",
    "nature": "Nature",
    "children": "Children",
    "adventure": "Adventure",
    "space": "Space",
    "film": "Film",
    "drama": "Drama",
    "documentary": "Documentary",
    "travel": "Travel",
    "sport": "Sport",
    "action": "Action",
    "epic": "Epic",
    "fairy": "Fairy Tale",
    "advertising": "Commercial",
    "commercial": "Commercial",
    "corporate": "Corporate",
    "background": "Background",
    "ballad": "Ballad",
    "party": "Party",
    "summer": "Summer",
    "retro": "Retro",
    "medieval": "Medieval",
    "trailer": "Trailer",
    "dramatic": "Drama",
    # ── 新补：原来未映射的标签 ──
    "movie": "Film",              # 413次，与 film 合并
    "game": "Game",               # 261次，新增 Theme
    "soundscape": "Ambient",      # 480次，映射到 Ambient 主题
    "horror": "Horror",           # 158次，新增 Theme
}

# MTG mood/theme 标签到系统 Scenario 节点的映射
# 右侧值即 Neo4j Scenario 节点名
# SCENARIO_TAG_MAP 中的英文关键词需要能 CONTAINS 匹配这些节点名
MTG_TO_SCENARIO = {
    "background": "Background Music",
    "film": "Film Soundtrack",
    "documentary": "Documentary",
    "advertising": "Advertising",
    "commercial": "Commercial",
    "corporate": "Corporate",
    "trailer": "Movie Trailer",
    "action": "Action Scene",
    "sport": "Workout",           # 与 SCENARIO_TAG_MAP "运动"→["workout"] 对齐
    "party": "Party",
    "children": "Children",
    "christmas": "Christmas",
    "holiday": "Holiday",
    "travel": "Travel",
    "meditative": "Relaxation",   # 与 SCENARIO_TAG_MAP "冥想"→["relaxing"] 对齐
    "ambiental": "Ambient",
    "game": "Gaming",             # 新增
}


def parse_moodtheme_tsv(tsv_path: str) -> Dict[str, dict]:
    """
    解析 autotagging_moodtheme.tsv
    
    TSV 格式:
      TRACK_ID  ARTIST_ID  ALBUM_ID  PATH  DURATION  TAG1,TAG2,...
      
    Returns:
      {track_id: {artist_id, album_id, path, duration, tags: [...]}}
    """
    tracks = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # 跳过表头
        
        for row in reader:
            if len(row) < 6:
                continue
            
            track_id = row[0]  # e.g., "track_0000948"
            artist_id = row[1]
            album_id = row[2]
            path = row[3]       # e.g., "48/948.mp3"
            duration = float(row[4]) if row[4] else 0.0
            
            # 标签可能跨多列（如果标签名里包含 tab）或以逗号分隔
            tags_raw = ",".join(row[5:])
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            
            tracks[track_id] = {
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "path": path,
                "duration": duration,
                "tags": tags,
            }
    
    return tracks


def parse_raw_meta_tsv(tsv_path: str) -> Dict[str, dict]:
    """
    解析 raw.meta.tsv（含 artist 名称、专辑名、曲名等）
    
    TSV 格式:
      TRACK_ID  ARTIST_ID  ALBUM_ID  TITLE  ARTIST_NAME  ALBUM_NAME  DATE  URL
      
    Returns:
      {track_id: {title, artist_name, album_name, release_date, url}}
    """
    meta = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        
        for row in reader:
            if len(row) < 6:
                continue
            
            track_id = row[0]
            meta[track_id] = {
                "title": row[3] if len(row) > 3 else "Unknown",
                "artist_name": row[4] if len(row) > 4 else "Unknown",
                "album_name": row[5] if len(row) > 5 else "Unknown",
                "release_date": row[6] if len(row) > 6 else "",
                "url": row[7] if len(row) > 7 else "",
            }
    
    return meta


def map_mtg_tags(tags: List[str]) -> dict:
    """
    将 MTG 原始标签映射到系统的 Mood/Theme/Scenario 分类
    
    输入: ["mood/theme---happy", "mood/theme---love", ...]
    输出: {moods: [...], themes: [...], scenarios: [...], vibe: "..."}
    """
    moods = []
    themes = []
    scenarios = []
    
    for tag in tags:
        # 解析 "mood/theme---happy" → "happy"
        if "---" in tag:
            tag_name = tag.split("---")[1].strip().lower()
        else:
            tag_name = tag.strip().lower()
        
        # 按优先级映射：同一个标签可能同时属于多个类别
        if tag_name in MTG_TO_MOOD:
            moods.append(MTG_TO_MOOD[tag_name])
        if tag_name in MTG_TO_THEME:
            themes.append(MTG_TO_THEME[tag_name])
        if tag_name in MTG_TO_SCENARIO:
            scenarios.append(MTG_TO_SCENARIO[tag_name])
    
    # 推断 vibe（综合利用 MTG 的描述性标签）
    vibe_parts = []
    tag_names = [t.split("---")[1] if "---" in t else t for t in tags]
    if any(k in tag_names for k in ["ambiental", "dream", "deep", "soundscape", "meditative"]):
        vibe_parts.append("Ambient")
    if any(k in tag_names for k in ["energetic", "action", "powerful", "upbeat", "fast", "heavy"]):
        vibe_parts.append("Energetic")
    if any(k in tag_names for k in ["calm", "relaxing", "soft", "tender", "mellow", "slow"]):
        vibe_parts.append("Acoustic")
    if any(k in tag_names for k in ["dark", "eerie", "aggressive", "horror"]):
        vibe_parts.append("Dark")
    if any(k in tag_names for k in ["happy", "funny", "cool", "groovy", "fun"]):
        vibe_parts.append("Upbeat")
    if any(k in tag_names for k in ["film", "epic", "dramatic", "trailer", "movie", "game"]):
        vibe_parts.append("Cinematic")
    if any(k in tag_names for k in ["melodic", "romantic", "ballad"]):
        vibe_parts.append("Melodic")
    # 取前 2 个 vibe 词拼接（如 "Ambient Melodic"）
    vibe = " ".join(vibe_parts[:2]) if vibe_parts else ""
    
    return {
        "moods": list(set(moods)),
        "themes": list(set(themes)),
        "scenarios": list(set(scenarios)),
        "vibe": vibe,
    }


def generate_metadata(
    tracks: Dict[str, dict],
    raw_meta: Dict[str, dict],
    output_dir: str,
    audio_source_dir: Optional[str] = None,
    sample_n: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[dict], set]:
    """
    生成与 ingest_to_neo4j.py 兼容的 _meta.json 文件，并可选地从源目录复制音频。
    
    关键设计：
      元数据文件名使用 "{track_id}.low" 作为基名（如 948.low_meta.json），
      与实际音频文件 948.low.mp3 对齐。
      ingest_to_neo4j.py 会 splitext("948.low.mp3") → basename="948.low"，
      然后查找 metadata/948.low_meta.json，完美匹配。
    
    Args:
        audio_source_dir: 已下载的 MTG 音频根目录（如 D:/mtg_training），
                          含 {folder_id}/{track_id}.low.mp3 结构。
                          如果提供，则自动复制采样到的文件到 output_dir/audio/。
    Returns:
        (track_list, required_folders)
    """
    # 随机采样
    track_ids = list(tracks.keys())
    if sample_n and sample_n < len(track_ids):
        random.seed(seed)
        track_ids = random.sample(track_ids, sample_n)
    
    # 确保输出目录存在
    metadata_dir = os.path.join(output_dir, "metadata")
    audio_out_dir = os.path.join(output_dir, "audio")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    
    track_list = []
    required_folders = set()
    copied_count = 0
    skipped_count = 0
    missing_count = 0
    
    for idx, track_id in enumerate(track_ids, 1):
        track = tracks[track_id]
        meta = raw_meta.get(track_id, {})
        
        # 提取数字 ID（从 "track_0000948" → "948"）
        numeric_id = track_id.replace("track_", "").lstrip("0") or "0"
        
        # 解析文件路径 "48/948.mp3" → folder=48, filename=948.mp3
        path_parts = track["path"].split("/")
        folder_id = path_parts[0] if len(path_parts) > 1 else ""
        audio_filename_original = path_parts[-1]  # e.g., "948.mp3"
        required_folders.add(folder_id)
        
        # 实际下载的文件名：{track_id}.low.mp3（如 948.low.mp3）
        audio_filename_on_disk = audio_filename_original.replace(".mp3", ".low.mp3")
        # 基名 = "948.low"（与 splitext 结果对齐）
        basename = os.path.splitext(audio_filename_on_disk)[0]  # "948.low"
        
        # 构建标签
        mapped = map_mtg_tags(track["tags"])
        
        # 元信息
        title = meta.get("title", f"MTG Track {numeric_id}")
        artist = meta.get("artist_name", "MTG-Jamendo Artist")
        album = meta.get("album_name", "MTG-Jamendo Album")
        
        # 构建 _meta.json（兼容 ingest_to_neo4j.py 格式）
        meta_data = {
            "musicId": f"mtg_{numeric_id}",
            "musicName": title,
            "artist": [[artist, 0]],
            "album": album,
            "duration": int(track["duration"] * 1000),
            "format": "mp3",
            "source": "mtg",
            "dataset": "mtg",
            "language": "English",
            "region": "Western",
            # MTG 映射后的标签（直接写入，供 ingest 脚本读取）
            "moods": mapped["moods"],
            "themes": mapped["themes"],
            "scenarios": mapped["scenarios"],
            "vibe": mapped["vibe"],
            # 原始 MTG 标签（保留用于溯源）
            "mtg_tags": track["tags"],
            "mtg_track_id": track_id,
            "mtg_path": track["path"],
        }
        
        # 写入 _meta.json（基名对齐：948.low_meta.json）
        meta_path = os.path.join(metadata_dir, f"{basename}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        # 复制音频文件（如果指定了源目录）
        if audio_source_dir:
            src = os.path.join(audio_source_dir, folder_id, audio_filename_on_disk)
            dst = os.path.join(audio_out_dir, audio_filename_on_disk)
            if os.path.exists(dst):
                skipped_count += 1
            elif os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"  ⚠️ 音频文件不存在: {src}")
            
            if idx % 500 == 0:
                print(f"  📦 进度 [{idx}/{len(track_ids)}] 已复制 {copied_count}, 跳过 {skipped_count}, 缺失 {missing_count}")
        
        track_list.append({
            "track_id": track_id,
            "numeric_id": numeric_id,
            "mtg_path": track["path"],
            "folder_id": folder_id,
            "audio_filename": audio_filename_on_disk,
            "basename": basename,
            "title": title,
            "artist": artist,
            **mapped,
        })
    
    if audio_source_dir:
        print(f"\n📊 音频复制统计: 复制 {copied_count}, 已存在跳过 {skipped_count}, 缺失 {missing_count}")
    
    return track_list, required_folders


def generate_gemini_result_json(track_list: List[dict], output_dir: str):
    """
    生成与 ingest_to_neo4j.py 兼容的 gemini_result.json 格式
    （注意：由于 MTG 适配器已将标签直接写入 _meta.json，
     ingest_to_neo4j.py 会优先从 meta.json 读取标签，
     因此这个文件主要作为备份/调试用途）
    """
    gemini_results = []
    for track in track_list:
        gemini_results.append({
            "filename": f"{track['basename']}.lrc",
            "moods": track.get("moods", []),
            "themes": track.get("themes", []),
            "scenarios": track.get("scenarios", []),
            "vibe": track.get("vibe", ""),
            "language": "English",
            "region": "Western",
        })
    
    out_path = os.path.join(output_dir, "gemini_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gemini_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 生成 gemini_result.json: {len(gemini_results)} 条 → {out_path}")


def print_summary(
    required_folders: set,
    track_list: List[dict],
    output_dir: str,
    has_audio_source: bool = False,
):
    """输出采样结果摘要"""
    print("\n" + "=" * 60)
    print(f"📋 采样完成！共 {len(track_list)} 首歌分布在 {len(required_folders)} 个文件夹中")
    print("=" * 60)
    
    # 标签统计
    print(f"\n📊 标签统计:")
    all_moods = defaultdict(int)
    all_themes = defaultdict(int)
    for track in track_list:
        for m in track.get("moods", []):
            all_moods[m] += 1
        for t in track.get("themes", []):
            all_themes[t] += 1
    
    print(f"   Mood  标签 TOP 10:")
    for mood, count in sorted(all_moods.items(), key=lambda x: -x[1])[:10]:
        print(f"     {mood:<20s} {count} 首")
    
    print(f"   Theme 标签 TOP 10:")
    for theme, count in sorted(all_themes.items(), key=lambda x: -x[1])[:10]:
        print(f"     {theme:<20s} {count} 首")
    
    print(f"\n📂 输出目录:")
    print(f"   元数据: {os.path.join(output_dir, 'metadata')}")
    print(f"   音频:   {os.path.join(output_dir, 'audio')}")
    
    # 下一步命令
    print(f"\n💡 下一步：运行以下命令导入 Neo4j:")
    print(f"   python data/pipeline/ingest_to_neo4j.py `")
    print(f"     --data-dir \"{os.path.join(output_dir, 'audio')}\" `")
    print(f"     --meta-dir \"{os.path.join(output_dir, 'metadata')}\" `")
    print(f"     --dataset mtg --skip-embeddings --force")


def main():
    parser = argparse.ArgumentParser(description="MTG-Jamendo 数据集适配器")
    parser.add_argument(
        "--tsv", type=str,
        default=str(MTG_METADATA_DIR / "autotagging_moodtheme.tsv"),
        help="autotagging_moodtheme.tsv 文件路径"
    )
    parser.add_argument(
        "--meta", type=str,
        default=str(MTG_METADATA_DIR / "raw.meta.tsv"),
        help="raw.meta.tsv 文件路径"
    )
    parser.add_argument(
        "--sample", type=int, default=4000,
        help="随机采样数量（默认 4000）"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="不采样，处理全部歌曲"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT.parent.parent / "data" / "mtg_sample"),
        help="输出目录（默认在 music_recommendation/data/mtg_sample，位于 git 仓库外）"
    )
    parser.add_argument(
        "--audio-source", type=str, default=None,
        help="已下载的 MTG 音频根目录（如 D:/mtg_training），"
             "含 {folder_id}/{track_id}.low.mp3 结构。"
             "如果提供则自动复制采样歌曲的音频到输出目录。"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 MTG-Jamendo 数据集适配器")
    print("=" * 60)
    
    # 1. 解析标签 TSV
    print(f"\n📖 解析标签文件: {args.tsv}")
    tracks = parse_moodtheme_tsv(args.tsv)
    print(f"   找到 {len(tracks)} 首带 mood/theme 标签的歌曲")
    
    # 2. 解析元信息 TSV
    print(f"\n📖 解析元信息文件: {args.meta}")
    raw_meta = parse_raw_meta_tsv(args.meta)
    print(f"   找到 {len(raw_meta)} 首歌的标题/歌手信息")
    
    # 3. 生成元数据 + 复制音频
    sample_n = None if args.all else args.sample
    if sample_n:
        print(f"\n🎲 随机采样 {sample_n} 首（seed={args.seed}）...")
    else:
        print(f"\n📦 处理全部 {len(tracks)} 首...")
    
    if args.audio_source:
        print(f"📂 音频源目录: {args.audio_source}")
        print(f"   将自动复制采样歌曲到: {os.path.join(args.output, 'audio')}")
    
    track_list, required_folders = generate_metadata(
        tracks, raw_meta, args.output,
        audio_source_dir=args.audio_source,
        sample_n=sample_n, seed=args.seed,
    )
    
    # 4. 生成兼容的 gemini_result.json（备份用）
    generate_gemini_result_json(track_list, args.output)
    
    # 5. 输出摘要
    print_summary(required_folders, track_list, args.output,
                  has_audio_source=args.audio_source is not None)


if __name__ == "__main__":
    main()
