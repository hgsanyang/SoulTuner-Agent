# ============================================================
# 【大统一入库脚本】Neo4j 图谱全量写入 (v2 — 并行 + 断点续传)
#
# 功能：
#   1. 读取 metadata/*.json (网易云原数据)
#   2. 读取 gemini_result.json (LLM 歌词标签)
#   3. 调用 M2D-CLAP + OMAR-RQ 提取双模型向量(可 skip)
#   4. 组装 Cypher MERGE 写入 Neo4j
#
# v2 新增：
#   - 断点续传：ingest_progress.json 记录 processing/done 状态
#   - 多线程写入：--skip-embeddings 模式下 8 线程并发写 Neo4j
#   - GPU batch 预加载：向量模式下多线程预加载音频再逐首推理
#   - --force / --reset-progress 参数
#
# 用法：
#   # 只写入元数据 + Gemini 标签（秒级完成，无需 GPU，8 线程并发）
#   python data_pipeline/ingest_to_neo4j.py --skip-embeddings
#
#   # 之后单独补充向量（需要 GPU，耗时较长）
#   python data_pipeline/ingest_to_neo4j.py --update-embeddings
#
#   # 一步到位：元数据 + 标签 + 向量 全部写入
#   python data_pipeline/ingest_to_neo4j.py
#
#   # 中途中断后重跑 → 自动跳过已完成的歌曲
#   python data_pipeline/ingest_to_neo4j.py --skip-embeddings
#
#   # 强制全量重跑（忽略进度文件）
#   python data_pipeline/ingest_to_neo4j.py --skip-embeddings --force
#
#   # 清空进度文件
#   python data_pipeline/ingest_to_neo4j.py --reset-progress
# ============================================================

import os
import sys
import json
import glob
import logging
import argparse
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- 默认目录配置（可通过 CLI 参数覆盖） ----
DEFAULT_DATA_ROOT = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio"
DEFAULT_AUDIO_DIR = os.path.join(DEFAULT_DATA_ROOT, "audio")
DEFAULT_COVER_DIR = os.path.join(DEFAULT_DATA_ROOT, "covers")
DEFAULT_LYRICS_DIR = os.path.join(DEFAULT_DATA_ROOT, "lyrics")
DEFAULT_METADATA_DIR = os.path.join(DEFAULT_DATA_ROOT, "metadata")
DEFAULT_GEMINI_RESULT_PATH = os.path.join(
    str(PROJECT_ROOT), "data_pipeline", "gemini_prompts", "gemini_result.json"
)

# 默认数据集标签
DEFAULT_DATASET = "personal"

# 进度文件：断点续传用（与脚本同目录）
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_progress.json")

# 多线程写入的并发数（--skip-embeddings 模式下使用）
MAX_WORKERS = 8

# 向量提取时的音频最大秒数（超过此长度只取前 N 秒）
# 原因：30分钟白噪音等超长音频会导致 OOM；模型在短片段上训练，
#       超长音频的 mean pooling 向量质量反而差。截取前 5 分钟足以表征风格。
MAX_AUDIO_SECONDS = 300  # 5 分钟

# 静态资源的 URL 前缀（前端访问时用）
STATIC_URL_PREFIX = "/static"


class ProgressTracker:
    """
    线程安全的进度追踪器。    使用三级状态防止中断时的残留问题：
      - processing:  标记已开始但未完成 → 重启时会被重新处理      - done_meta:   元数据/标签已写入，但还没有向量 → --update-embeddings 时重新处理      - done_full:   元数据+标签+向量 全部写入 → 所有模式下都跳过    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._progress: Dict[str, str] = {}  # {music_id: "processing"|"done_meta"|"done_full"}
        self._load()

    def _load(self):
        """从磁盘加载进度"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self._progress = json.load(f)
                # 兼容旧版 "done" 状态 → 视为 "done_meta"
                for k, v in self._progress.items():
                    if v == "done":
                        self._progress[k] = "done_meta"
                meta = sum(1 for v in self._progress.values() if v == "done_meta")
                full = sum(1 for v in self._progress.values() if v == "done_full")
                proc = self._count_processing()
                logger.info(f"📋 加载进度文件: {meta} 首仅标签, {full} 首含向量, "
                            f"{proc} 首处理中(将重试)")
            except Exception as e:
                logger.warning(f"进度文件损坏，将重新开始: {e}")
                self._progress = {}

    def _save(self):
        """写入磁盘（已在锁内调用）"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self._progress, f, ensure_ascii=False, indent=2)

    def _count_processing(self) -> int:
        return sum(1 for v in self._progress.values() if v == "processing")

    def should_skip(self, music_id: str, need_embeddings: bool) -> bool:
        """
        判断是否应该跳过这首歌：
          - need_embeddings=False (--skip-embeddings): done_meta 或 done_full 都跳过          - need_embeddings=True  (默认/--update-embeddings): 只有 done_full 才跳过        """
        with self._lock:
            status = self._progress.get(music_id)
            if status == "done_full":
                return True  # 已完整入库（含向量），任何模式都跳过
            if status == "done_meta" and not need_embeddings:
                return True  # 仅标签入库 + 当前模式不需要向量 → 跳过
            return False

    def mark_processing(self, music_id: str):
        """标记为处理中"""
        with self._lock:
            self._progress[music_id] = "processing"
            self._save()

    def mark_done(self, music_id: str, with_embeddings: bool):
        """标记为已完成"""
        with self._lock:
            self._progress[music_id] = "done_full" if with_embeddings else "done_meta"
            self._save()

    def reset(self):
        """清空进度"""
        self._progress = {}
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        logger.info("🗑️ 进度文件已清理")

    @property
    def done_count(self) -> int:
        with self._lock:
            return sum(1 for v in self._progress.values() if v.startswith("done"))


class UnifiedIngestion:
    """大统一入库器：元数据 + Gemini标签 + M2D-CLAP/OMAR-RQ向量 → Neo4j"""

    def __init__(self, force: bool = False, dataset: str = DEFAULT_DATASET,
                 audio_dir: str = None, metadata_dir: str = None,
                 gemini_result_path: str = None):
        self.client = get_neo4j_client()
        self.dataset = dataset
        self.audio_dir = audio_dir or DEFAULT_AUDIO_DIR
        self.metadata_dir = metadata_dir or DEFAULT_METADATA_DIR
        self.gemini_result_path = gemini_result_path or DEFAULT_GEMINI_RESULT_PATH
        self.gemini_tags = self._load_gemini_tags()
        self._embedder_loaded = False
        self.progress = ProgressTracker(PROGRESS_FILE)
        self.force = force

    def _load_gemini_tags(self) -> Dict[str, Dict]:
        """加载 Gemini 提取的歌词标签，以 filename 为 key 建立索引"""
        tags_index = {}
        if not os.path.exists(self.gemini_result_path):
            logger.warning(f"Gemini 标签文件不存在: {self.gemini_result_path}")
            return tags_index
        try:
            with open(self.gemini_result_path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                # 自动清洗尾部非法字符（如 Gemini 可能返回的 ]. 或 ]。）
                while raw and raw[-1] not in ']':
                    raw = raw[:-1]
                data = json.loads(raw)
            for item in data:
                fn = item.get("filename", "")
                if fn:
                    tags_index[fn] = item
            logger.info(f"✅ 加载 Gemini 标签: {len(tags_index)} 条")
        except Exception as e:
            logger.error(f"加载 Gemini 标签失败: {e}")
        return tags_index

    def _ensure_embedder(self):
        """懒加载 M2D-CLAP + OMAR-RQ 模型"""
        if self._embedder_loaded:
            return
        from retrieval.audio_embedder import get_m2d2_model, get_omar_model
        logger.info("正在加载 M2D-CLAP 跨模态模型..")
        get_m2d2_model()
        logger.info("正在加载 OMAR-RQ (multicodebook) 音频特征模型...")
        get_omar_model()
        self._embedder_loaded = True
        logger.info("✅ 双模型加载完毕")

    def _extract_embeddings(self, audio_path: str) -> Dict[str, List[float]]:
        """
        对单个音频文件提取双模型向量（M2D-CLAP + OMAR-RQ 均使用 16kHz）。        超过 MAX_AUDIO_SECONDS 的音频只截取前 N 秒，防止 OOM。        """
        import librosa
        from retrieval.audio_embedder import encode_audio_to_embedding, extract_audio_representation

        # 先快速读取时长（不解码音频数据，几乎零开销）
        file_duration = librosa.get_duration(path=audio_path)

        # 超长音频只加载前 MAX_AUDIO_SECONDS 秒
        load_duration = None
        if file_duration > MAX_AUDIO_SECONDS:
            load_duration = MAX_AUDIO_SECONDS
            logger.info(f"  ⏱️ 音频时长 {file_duration:.0f}s 超过上限，只截取前 {MAX_AUDIO_SECONDS}s 提取向量")

        audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)

        # 两个模型统一使用 16kHz，只需重采样一次
        audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        # M2D-CLAP: 跨模态向量（文本-音频同空间）
        m2d2_emb = encode_audio_to_embedding(audio_16k, sample_rate=16000)

        # OMAR-RQ: 纯音频特征向量（声学表征）
        omar_emb = extract_audio_representation(audio_16k, sample_rate=16000)

        return {"m2d2_embedding": m2d2_emb, "omar_embedding": omar_emb}

    def _quick_music_id(self, audio_path: str) -> str:
        """快速获取 music_id（用于启动时统计，不加载完整元数据）"""
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        meta = self._load_metadata(basename)
        if meta:
            return str(meta.get("musicId", f"local_{basename}"))
        return f"local_{basename}"

    def _preload_audio(self, audio_path: str):
        """预加载并重采样音频到 16kHz（用于多线程预加载）"""
        import librosa
        file_duration = librosa.get_duration(path=audio_path)
        load_duration = MAX_AUDIO_SECONDS if file_duration > MAX_AUDIO_SECONDS else None
        audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)
        audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        return audio_16k

    def _load_metadata(self, basename: str) -> Optional[Dict]:
        """读取对应的 _meta.json"""
        meta_path = os.path.join(self.metadata_dir, f"{basename}_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _get_gemini_tags(self, basename: str) -> Dict:
        """查找对应的 Gemini 标签"""
        lrc_key = f"{basename}.lrc"
        return self.gemini_tags.get(lrc_key, {})

    def _build_static_urls(self, basename: str, ext: str) -> Dict[str, str]:
        """构建前端可用的静态资源 URL（根据数据集选择不同路径）"""
        if self.dataset == "mtg":
            return {
                "audio_url": f"/static/mtg_audio/{basename}.{ext}",
                "cover_url": "",   # MTG 无封面
                "lrc_url": "",     # MTG 无歌词
            }
        return {
            "audio_url": f"{STATIC_URL_PREFIX}/audio/{basename}.{ext}",
            "cover_url": f"{STATIC_URL_PREFIX}/covers/{basename}_cover.jpg",
            "lrc_url": f"{STATIC_URL_PREFIX}/lyrics/{basename}.lrc",
        }

    def _write_song_to_neo4j(self, song_data: Dict[str, Any]):
        """
        以 MERGE 写入一首歌的完整数据到 Neo4j。        包括：Song节点 + Artist关系 + Mood/Theme/Scenario 标签节点和关系。        """
        # ---- 核心 Song MERGE ----
        set_parts = [
            "s.title = $title",
            "s.album = $album",
            "s.duration = $duration",
            "s.format = $format",
            "s.audio_url = $audio_url",
            "s.cover_url = $cover_url",
            "s.lrc_url = $lrc_url",
            "s.vibe = $vibe",
            "s.language = $language",
            "s.region = $region",
            "s.dataset = $dataset",
            "s.updated_at = timestamp()",
        ]
        params = {
            "music_id": song_data["music_id"],
            "title": song_data["title"],
            "artist_name": song_data["artist"],
            "album": song_data.get("album", "Unknown"),
            "duration": song_data.get("duration", 0),
            "format": song_data.get("format", "flac"),
            "audio_url": song_data.get("audio_url", ""),
            "cover_url": song_data.get("cover_url", ""),
            "lrc_url": song_data.get("lrc_url", ""),
            "vibe": song_data.get("vibe", ""),
            "moods": song_data.get("moods", []),
            "themes": song_data.get("themes", []),
            "scenarios": song_data.get("scenarios", []),
            "genres": song_data.get("genres", []),
            "language": song_data.get("language", ""),
            "region": song_data.get("region", ""),
            "dataset": song_data.get("dataset", self.dataset),
        }

        # 向量字段（可选）
        if song_data.get("m2d2_embedding"):
            set_parts.append("s.m2d2_embedding = $m2d2_embedding")
            params["m2d2_embedding"] = song_data["m2d2_embedding"]
        if song_data.get("omar_embedding"):
            set_parts.append("s.omar_embedding = $omar_embedding")
            params["omar_embedding"] = song_data["omar_embedding"]

        set_clause = ", ".join(set_parts)

        query = f"""
        MERGE (s:Song {{music_id: $music_id}})
        SET {set_clause}

        MERGE (a:Artist {{name: $artist_name}})
        MERGE (s)-[:PERFORMED_BY]->(a)

        WITH s
        OPTIONAL MATCH (s)-[r_m:HAS_MOOD]->()    DELETE r_m
        WITH s
        OPTIONAL MATCH (s)-[r_t:HAS_THEME]->()   DELETE r_t
        WITH s
        OPTIONAL MATCH (s)-[r_s:FITS_SCENARIO]->() DELETE r_s
        WITH s
        OPTIONAL MATCH (s)-[r_l:HAS_LANGUAGE]->() DELETE r_l
        WITH s
        OPTIONAL MATCH (s)-[r_r:IN_REGION]->()    DELETE r_r
        WITH s
        OPTIONAL MATCH (s)-[r_g:BELONGS_TO_GENRE]->() DELETE r_g

        WITH s
        FOREACH (mood IN $moods |
            MERGE (m:Mood {{name: mood}})
            MERGE (s)-[:HAS_MOOD]->(m)
        )

        WITH s
        FOREACH (theme IN $themes |
            MERGE (t:Theme {{name: theme}})
            MERGE (s)-[:HAS_THEME]->(t)
        )

        WITH s
        FOREACH (scenario IN $scenarios |
            MERGE (sc:Scenario {{name: scenario}})
            MERGE (s)-[:FITS_SCENARIO]->(sc)
        )

        WITH s
        FOREACH (genre IN $genres |
            MERGE (g:Genre {{name: genre}})
            MERGE (s)-[:BELONGS_TO_GENRE]->(g)
        )

        WITH s
        FOREACH (_ IN CASE WHEN $language <> '' THEN [1] ELSE [] END |
            MERGE (lang:Language {{name: $language}})
            MERGE (s)-[:HAS_LANGUAGE]->(lang)
        )

        WITH s
        FOREACH (_ IN CASE WHEN $region <> '' THEN [1] ELSE [] END |
            MERGE (reg:Region {{name: $region}})
            MERGE (s)-[:IN_REGION]->(reg)
        )
        """
        self.client.execute_query(query, params)

    def _prepare_song_data(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        准备单首歌的完整数据（不含向量）。        返回 song_data dict，或 None（如果无法处理）。        """
        filename = os.path.basename(audio_path)
        basename, ext = os.path.splitext(filename)
        ext = ext.lstrip(".")

        # 1. 读取元数据（兼容网易云格式和 MTG 格式）
        meta = self._load_metadata(basename) or {}
        music_id = str(meta.get("musicId", f"local_{basename}"))
        title = meta.get("musicName", basename.split(" - ")[0] if " - " in basename else basename)
        artists_raw = meta.get("artist", [])
        artist = "、".join([a[0] for a in artists_raw]) if artists_raw else (
            basename.split(" - ")[1] if " - " in basename else "Unknown"
        )
        album = meta.get("album", "Unknown")
        duration = meta.get("duration", 0)
        fmt = meta.get("format", ext)
        dataset = meta.get("dataset", self.dataset)

        # 2. 读取标签：优先从 meta.json 中读取（MTG 适配器已写入），否则用 Gemini 标签
        if meta.get("moods") or meta.get("themes") or meta.get("scenarios"):
            # MTG 适配器已将标签写入 _meta.json
            moods = meta.get("moods", [])
            themes = meta.get("themes", [])
            scenarios = meta.get("scenarios", [])
            vibe = meta.get("vibe", "")
            language = meta.get("language", "English")
            region = meta.get("region", "")
            genres = meta.get("genres", [])  # MTG 通常无此字段
        else:
            # 使用 Gemini 标签（个人音乐入库路径）
            tags = self._get_gemini_tags(basename)
            moods = tags.get("moods", [])
            themes = tags.get("themes", [])
            scenarios = tags.get("scenarios", [])
            vibe = tags.get("vibe", "")
            language = tags.get("language", "")
            region = tags.get("region", "")
            # genre 可能是字符串或列表（兼容旧版 JSON）
            raw_genre = tags.get("genre", tags.get("genres", []))
            if isinstance(raw_genre, str):
                genres = [raw_genre] if raw_genre else []
            else:
                genres = raw_genre or []

        # 3. 构建静态 URL
        urls = self._build_static_urls(basename, ext)

        return {
            "music_id": music_id,
            "title": title,
            "artist": artist,
            "album": album,
            "duration": duration,
            "format": fmt,
            "dataset": dataset,
            "moods": moods,
            "themes": themes,
            "scenarios": scenarios,
            "genres": genres,
            "vibe": vibe,
            "language": language,
            "region": region,
            "m2d2_embedding": None,
            "omar_embedding": None,
            "audio_path": audio_path,
            **urls,
        }

    def _process_one_song(self, audio_path: str, skip_embeddings: bool,
                          idx: int, total: int) -> bool:
        """
        处理单首歌的完整流程（线程安全）。        返回 True 表示成功，False 表示失败。        """
        filename = os.path.basename(audio_path)
        basename = os.path.splitext(filename)[0]
        need_embeddings = not skip_embeddings

        try:
            # 准备数据
            song_data = self._prepare_song_data(audio_path)
            if not song_data:
                logger.warning(f"  ⚠️ 无法准备数据: {basename}")
                return False

            music_id = song_data["music_id"]

            # 断点续传检查（区分 done_meta / done_full）
            if not self.force and self.progress.should_skip(music_id, need_embeddings):
                logger.info(f"  ⏭️ [{idx}/{total}] 已完成，跳过: {song_data['title']} - {song_data['artist']}")
                return True

            # 标记为 processing（如果中断，重启时会重试这首）
            self.progress.mark_processing(music_id)

            # 提取向量（可选，非线程安全操作，仅在串行模式下调用）
            if need_embeddings:
                logger.info(f"  🧠 [{idx}/{total}] 提取 M2D2 + OMAR 向量: {basename}")
                embs = self._extract_embeddings(audio_path)
                song_data["m2d2_embedding"] = embs["m2d2_embedding"]
                song_data["omar_embedding"] = embs["omar_embedding"]
                logger.info(f"  ✅ M2D2: {len(embs['m2d2_embedding'])}维, OMAR: {len(embs['omar_embedding'])}维")

            # 写入 Neo4j
            self._write_song_to_neo4j(song_data)

            # 标记为 done（区分是否包含向量）
            self.progress.mark_done(music_id, with_embeddings=need_embeddings)
            logger.info(f"  ✅ [{idx}/{total}] 入库成功: {song_data['title']} - {song_data['artist']}")
            return True

        except Exception as e:
            logger.error(f"  ❌ [{idx}/{total}] 失败 ({basename}): {e}")
            return False

    def ingest_all(self, skip_embeddings: bool = False, update_embeddings_only: bool = False):
        """
        主入口：扫描音频目录，写入 Neo4j。
        Args:
            skip_embeddings: True = 只写元数据/标签，跳过耗时的模型推理（多线程并发）
            update_embeddings_only: True = 只更新已入库歌曲的向量        """
        audio_files = []
        for ext in ("*.flac", "*.mp3", "*.wav", "*.ogg"):
            # 顶层目录扫描
            audio_files.extend(glob.glob(os.path.join(self.audio_dir, ext)))
            # 递归扫描子目录（用于 MTG 的 {folder_id}/{track_id}.mp3 结构）
            audio_files.extend(glob.glob(os.path.join(self.audio_dir, "**", ext), recursive=True))
        # 去重
        audio_files = list(set(audio_files))

        if not audio_files:
            logger.info(f"✅ {self.audio_dir} 中没有找到任何音频文件")
            return

        total = len(audio_files)
        need_embeddings = not skip_embeddings
        # 按当前模式统计：实际会被跳过的 vs 需要处理的
        if not self.force:
            will_skip = sum(1 for f in audio_files
                           if self.progress.should_skip(
                               self._quick_music_id(f), need_embeddings))
        else:
            will_skip = 0
        logger.info(f"🚀 找到 {total} 个音频文件，"
                    f"(本次将跳过: {will_skip}, 待处理: {total - will_skip})")

        if not skip_embeddings or update_embeddings_only:
            self._ensure_embedder()

        success = 0
        errors = 0
        skipped = 0

        if skip_embeddings:
            # ══════════════════════════════════════════════════════
            # 模式 A：多线程并发写入（无 GPU 依赖，I/O 密集型）
            # ══════════════════════════════════════════════════════
            logger.info(f"🧵 启动 {MAX_WORKERS} 线程并发写入模式 (--skip-embeddings)")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_path = {}
                for idx, audio_path in enumerate(audio_files, 1):
                    future = executor.submit(
                        self._process_one_song, audio_path, True, idx, total
                    )
                    future_to_path[future] = audio_path

                for future in as_completed(future_to_path):
                    try:
                        result = future.result()
                        if result:
                            success += 1
                        else:
                            errors += 1
                    except Exception as e:
                        errors += 1
                        path = future_to_path[future]
                        logger.error(f"  ❌ 线程异常 ({os.path.basename(path)}): {e}")
        else:
            # ══════════════════════════════════════════════════════
            # 模式 B：串行处理（GPU 推理是瓶颈，多线程无意义）
            # 但用后台线程预加载下一首的音频数据
            # ══════════════════════════════════════════════════════
            logger.info("🔥 启动 GPU 串行推理模式 (含向量提取")

            for idx, audio_path in enumerate(audio_files, 1):
                result = self._process_one_song(audio_path, False, idx, total)
                if result:
                    success += 1
                else:
                    errors += 1

        logger.info("=" * 60)
        logger.info(f"🎉 入库完成! 成功: {success}, 失败: {errors}")
        logger.info(f"📋 进度文件: {PROGRESS_FILE}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大统一 Neo4j 入库脚本 (v2 — 并行 + 断点续传 + 多数据集)")
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="跳过 M2D2/OMAR 向量提取（只写元数据+标签，8 线程并发，秒级完成）"
    )
    parser.add_argument(
        "--update-embeddings", action="store_true",
        help="仅更新已入库歌曲的向量（需要 GPU）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="忽略进度文件，强制全量重新处理"
    )
    parser.add_argument(
        "--reset-progress", action="store_true",
        help="清空进度文件后退出"
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help=f"数据集标签，写入 Song 节点的 dataset 属性 (默认: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="自定义音频数据目录（默认使用 processed_audio/audio）"
    )
    parser.add_argument(
        "--meta-dir", type=str, default=None,
        help="自定义元数据目录（默认使用 processed_audio/metadata）"
    )
    parser.add_argument(
        "--gemini-result", type=str, default=None,
        help="自定义 Gemini/标签结果 JSON 文件路径"
    )
    args = parser.parse_args()

    # 处理 --reset-progress
    if args.reset_progress:
        tracker = ProgressTracker(PROGRESS_FILE)
        tracker.reset()
        print("✅ 进度文件已清理")
        sys.exit(0)

    ingestion = UnifiedIngestion(
        force=args.force,
        dataset=args.dataset,
        audio_dir=args.data_dir,
        metadata_dir=args.meta_dir,
        gemini_result_path=args.gemini_result,
    )
    ingestion.ingest_all(
        skip_embeddings=args.skip_embeddings,
        update_embeddings_only=args.update_embeddings
    )
