# ============================================================
# 【V2 升级】数据飞轮 V2 ── 自动打标入库流水线# 来源：V2 架构重构方案 → Phase 3 / Step 7
#
# 替代旧版基于 SentenceTransformer + Milvus 的 data_flywheel.py
#
# 核心流程：
#   1. 扫描本地音频文件夹（或导入歌单名单）
#   2. OMAR-RQ/MERT 纯音频自动打标（本地推理，无需联网）
#   3. LLM 语义打标增强（调用 Gemini/Qwen 生成人文描述）
#   4. M2D2/CLAP 提取跨模态 Embedding
#   5. 全部写入 Neo4j Song 节点
#
# 这就是你的"私有音乐知识图谱的造血机器"！# ============================================================

import os
import sys
import json
import glob
import logging
import argparse
from typing import List, Dict, Any, Optional

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataFlywheelV2:
    """
    【V2 升级】数据飞轮 V2：双模型自动打标 + Neo4j 入库
    
    替代旧版基于 SentenceTransformer + Milvus 的 DataFlywheelPipeline。    现在直接与 Neo4j 图数据库交互，使用双模型提取特征。    """
    
    def __init__(self, watch_dir: str = "./data/new_audio"):
        self.watch_dir = watch_dir
        os.makedirs(self.watch_dir, exist_ok=True)
        
        # 懒加载标记        self._models_loaded = False
    
    def _ensure_models_loaded(self):
        """【V2 升级】懒加载双模型"""
        if self._models_loaded:
            return
        
        from retrieval.audio_embedder import get_m2d2_model, get_omar_model
        
        logger.info("[DataFlywheel V2] 正在加载 M2D2 跨模态模型..")
        get_m2d2_model()
        logger.info("[DataFlywheel V2] 正在加载 OMAR/MERT 音频特征模型...")
        get_omar_model()
        
        self._models_loaded = True
        logger.info("[DataFlywheel V2] ✅ 双模型加载完毕")
    
    def _extract_embeddings(self, audio_path: str) -> Dict[str, List[float]]:
        """
        【V2 升级】对单个音频提取双模型 Embedding
        """
        import librosa
        from retrieval.audio_embedder import encode_audio_to_embedding, extract_audio_representation
        
        # 加载音频（统一为单声道）        audio_np, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # M2D2: 跨模态 Embedding (需要 48kHz)
        audio_48k = librosa.resample(audio_np, orig_sr=sr, target_sr=48000)
        m2d2_emb = encode_audio_to_embedding(audio_48k, sample_rate=48000)
        
        # OMAR/MERT: 纯音频特征(需要 24kHz)
        audio_24k = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
        omar_emb = extract_audio_representation(audio_24k, sample_rate=24000)
        
        return {
            "m2d2_embedding": m2d2_emb,
            "omar_embedding": omar_emb
        }
    
    def _llm_auto_tag(self, song_name: str, artist: str = "") -> Dict[str, Any]:
        """
        【V2 升级】调用 LLM 生成语义标签（genre/mood/instrument/tempo/场景描述）        
        返回格式: {
            "genre": "Pop/R&B",
            "mood": "melancholy, nostalgic",
            "instruments": "piano, strings, soft drums",
            "tempo": "slow",
            "scene": "适合深夜独处或下雨天窗边发呆",
            "description": "一首融合了 R&B 和流行元素的抒情慢歌..."
        }
        """
        try:
            from llms.multi_llm import get_chat_model
            
            llm = get_chat_model()
            
            prompt = f"""你是一个专业的音乐标签分析师。根据以下歌曲信息，生成详细的音乐标签。            
歌曲名: {song_name}
歌手: {artist if artist else '未知'}

请严格按照以下 JSON 格式输出（不要输出任何其他内容）:
{{
    "genre": "流派(用英文，多个用逗号分隔)",
    "mood": "情绪标签(用英文，多个用逗号分隔)",
    "instruments": "主要乐器(用英文，多个用逗号分隔)",
    "tempo": "节奏(slow/medium/fast)",
    "scene": "适合的场景(用中文简短描述)",
    "description": "一句话音乐描述(用中文)"
}}"""
            
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*?\}', content)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"[DataFlywheel V2] LLM 打标失败 [{song_name}]: {e}")
        
        # 回退到空标签
        return {
            "genre": "Unknown",
            "mood": "neutral",
            "instruments": "",
            "tempo": "medium",
            "scene": "",
            "description": ""
        }
    
    def _write_to_neo4j(
        self,
        title: str,
        artist: str,
        m2d2_embedding: List[float],
        omar_embedding: List[float],
        auto_tags: Dict[str, Any],
        filepath: str = ""
    ):
        """
        【V2 升级】写入 Neo4j Song 节点（带双向量 + 标签）        """
        client = get_neo4j_client()
        
        genre = auto_tags.get("genre", "")
        
        query = """
        MERGE (s:Song {title: $title, artist: $artist_name})
        SET s.m2d2_embedding = $m2d2_embedding,
            s.omar_embedding = $omar_embedding,
            s.genre = $genre,
            s.mood = $mood,
            s.instruments = $instruments,
            s.tempo = $tempo,
            s.scene = $scene,
            s.description = $description,
            s.filepath = $filepath,
            s.updated_at = timestamp()
        
        MERGE (a:Artist {name: $artist_name})
        MERGE (s)-[:PERFORMED_BY]->(a)
        
        WITH s, $genre AS genre_name
        FOREACH (_ IN CASE WHEN genre_name <> '' THEN [1] ELSE [] END |
            MERGE (g:Genre {name: genre_name})
            MERGE (s)-[:BELONGS_TO_GENRE]->(g)
        )
        """
        
        params = {
            "title": title,
            "artist_name": artist,
            "m2d2_embedding": m2d2_embedding,
            "omar_embedding": omar_embedding,
            "genre": genre,
            "mood": auto_tags.get("mood", ""),
            "instruments": auto_tags.get("instruments", ""),
            "tempo": auto_tags.get("tempo", ""),
            "scene": auto_tags.get("scene", ""),
            "description": auto_tags.get("description", ""),
            "filepath": filepath,
        }
        
        client.execute_query(query, params)
    
    def ingest_audio_files(self, limit: Optional[int] = None):
        """
        【V2 升级】主入口：扫描音频文件夹 → 双模型提取 → LLM 打标 → Neo4j 入库
        """
        self._ensure_models_loaded()
        
        # 扫描支持的音频格式        supported_extensions = ('*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a')
        audio_files = []
        for ext in supported_extensions:
            audio_files.extend(glob.glob(os.path.join(self.watch_dir, '**', ext), recursive=True))
        
        if not audio_files:
            logger.info(f"[DataFlywheel V2] 未在 {self.watch_dir} 中发现音频文件")
            return
        
        if limit:
            audio_files = audio_files[:limit]
        
        logger.info(f"[DataFlywheel V2] 🚀 开始处理 {len(audio_files)} 个音频文件")
        
        success_count = 0
        error_count = 0
        
        for idx, file_path in enumerate(audio_files):
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            
            # 从文件名尝试解析 "歌手 - 歌名" 格式
            if " - " in title:
                parts = title.split(" - ", 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            else:
                artist = "Unknown"
            
            logger.info(f"[{idx+1}/{len(audio_files)}] 处理: {artist} - {title}")
            
            try:
                # Step 1: 双模型提取 Embedding
                embeddings = self._extract_embeddings(file_path)
                
                # Step 2: LLM 语义打标
                auto_tags = self._llm_auto_tag(title, artist)
                logger.info(f"  标签: {auto_tags.get('genre', '')} | {auto_tags.get('mood', '')}")
                
                # Step 3: 写入 Neo4j
                self._write_to_neo4j(
                    title=title,
                    artist=artist,
                    m2d2_embedding=embeddings["m2d2_embedding"],
                    omar_embedding=embeddings["omar_embedding"],
                    auto_tags=auto_tags,
                    filepath=file_path
                )
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ 失败: {e}")
                error_count += 1
        
        logger.info(f"[DataFlywheel V2] 🎉 处理完毕! 成功: {success_count}, 失败: {error_count}")
    
    def ingest_from_songlist(self, songs: List[Dict[str, str]]):
        """
        【V2 升级】从歌单列表导入（无音频文件时，仅做 LLM 打标 + 入库）        
        Args:
            songs: [{"title": "晴天", "artist": "周杰伦"}, ...]
        """
        logger.info(f"[DataFlywheel V2] 从歌单导入 {len(songs)} 首歌曲（仅标签，无向量）")
        
        for idx, song in enumerate(songs):
            title = song.get("title", "Unknown")
            artist = song.get("artist", "Unknown")
            
            logger.info(f"[{idx+1}/{len(songs)}] 打标: {artist} - {title}")
            
            try:
                auto_tags = self._llm_auto_tag(title, artist)
                
                # 无音频时，embedding 为空列表
                self._write_to_neo4j(
                    title=title,
                    artist=artist,
                    m2d2_embedding=[],
                    omar_embedding=[],
                    auto_tags=auto_tags
                )
            except Exception as e:
                logger.error(f"  ❌ 失败: {e}")
        
        logger.info(f"[DataFlywheel V2] ✅ 歌单导入完毕")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 数据飞轮 — 自动打标入库")
    parser.add_argument("--dir", type=str, default="./data/new_audio", help="音频文件目录")
    parser.add_argument("--limit", type=int, default=None, help="测试限制数量")
    args = parser.parse_args()
    
    flywheel = DataFlywheelV2(watch_dir=args.dir)
    flywheel.ingest_audio_files(limit=args.limit)
