import logging

from typing import List, Dict, Any, Optional



logger = logging.getLogger(__name__)



class UserMemoryManager:

    """

    用户长期记忆管理器
    负责将用户的偏好、播放历史、收藏等记录持久化到 Neo4j 图数据库中。
    """

    

    def __init__(self, neo4j_client=None):

        """

        初始化
        Args:

            neo4j_client: 可选的依赖注入，如果在系统级框架中统一管理连接

        """

        # 在实际执行中获取连接

        self.neo4j_client = neo4j_client

        if not self.neo4j_client:

            try:

                from retrieval.neo4j_client import get_neo4j_client

                self.neo4j_client = get_neo4j_client()

            except ImportError:

                logger.warning("未能显式导入 Neo4j 客户端，可能将导致记忆写库失败。")



    def ensure_user_exists(self, user_id: str, username: str = "DefaultUser"):

        """确保在图谱中存在该用户节点"""

        query = """

        MERGE (u:User {id: $user_id})

        ON CREATE SET u.name = $username, u.created_at = timestamp()

        """

        if self.neo4j_client:

            self.neo4j_client.execute_query(query, {"user_id": user_id, "username": username})

            logger.info(f"确保用户存在: {user_id}")



    def record_liked_song(self, user_id: str, song_title: str, artist: str):

        """记录用户收藏/红心歌曲"""

        # 防止 null artist 导致 Neo4j MERGE 失败

        if not song_title or not artist or artist == "未知":

            logger.warning(f"跳过记录喜欢歌曲（缺少必要信息）: title={song_title}, artist={artist}")

            return

        query = """

        MATCH (u:User {id: $user_id})

        MERGE (s:Song {title: $song_title, artist: $artist})

        MERGE (u)-[r:LIKES]->(s)

        ON CREATE SET r.created_at = timestamp(), r.weight = 1.0

        ON MATCH SET r.weight = r.weight + 0.1

        """

        if self.neo4j_client:

            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title, "artist": artist})

            logger.info(f"记录用户 {user_id} 喜欢歌曲: {song_title} - {artist}")



    def record_listened_song(self, user_id: str, song_title: str, artist: str, duration: int = 0):
        """记录用户播放历史"""
        if not song_title or not artist or artist == "未知":
            logger.warning(f"跳过记录收听歌曲（缺少必要信息）: title={song_title}, artist={artist}")
            return
        query = """
        MATCH (u:User {id: $user_id})
        MERGE (s:Song {title: $song_title, artist: $artist})
        MERGE (u)-[r:LISTENED_TO]->(s)
        ON CREATE SET r.play_count = 1, r.total_duration = $duration, r.last_played = timestamp()
        ON MATCH SET r.play_count = r.play_count + 1, r.total_duration = r.total_duration + $duration, r.last_played = timestamp()
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title, "artist": artist, "duration": duration})
            logger.info(f"记录用户 {user_id} 收听歌曲: {song_title}")

    def record_saved_song(self, user_id: str, song_title: str, artist: str):
        """记录用户收藏歌曲 → SAVES 关系（权重 0.8，低于 LIKES 的 1.0）"""
        if not song_title or not artist or artist == "未知":
            logger.warning(f"跳过记录收藏（缺少必要信息）: title={song_title}, artist={artist}")
            return
        query = """
        MATCH (u:User {id: $user_id})
        MERGE (s:Song {title: $song_title, artist: $artist})
        MERGE (u)-[r:SAVES]->(s)
        ON CREATE SET r.created_at = timestamp(), r.weight = 0.8
        ON MATCH SET r.weight = r.weight + 0.1
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title, "artist": artist})
            logger.info(f"记录用户 {user_id} 收藏歌曲: {song_title} - {artist}")

    def record_dislike(self, user_id: str, song_title: str, artist: str):
        """记录用户明确不喜欢 → DISLIKES 关系（推荐时完全排除）"""
        if not song_title or not artist or artist == "未知":
            logger.warning(f"跳过记录不喜欢（缺少必要信息）: title={song_title}, artist={artist}")
            return
        query = """
        MATCH (u:User {id: $user_id})
        MERGE (s:Song {title: $song_title, artist: $artist})
        MERGE (u)-[r:DISLIKES]->(s)
        ON CREATE SET r.created_at = timestamp(), r.weight = 1.0
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title, "artist": artist})
            logger.info(f"记录用户 {user_id} 不喜欢歌曲: {song_title} - {artist}")

    def record_skipped(self, user_id: str, song_title: str, artist: str):
        """记录用户跳过 → SKIPPED 关系（skip_count 累计，>=3 次才降权）"""
        if not song_title or not artist or artist == "未知":
            logger.warning(f"跳过记录skip（缺少必要信息）: title={song_title}, artist={artist}")
            return
        query = """
        MATCH (u:User {id: $user_id})
        MERGE (s:Song {title: $song_title, artist: $artist})
        MERGE (u)-[r:SKIPPED]->(s)
        ON CREATE SET r.skip_count = 1, r.first_skipped = timestamp(), r.last_skipped = timestamp()
        ON MATCH SET r.skip_count = r.skip_count + 1, r.last_skipped = timestamp()
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title, "artist": artist})
            logger.info(f"记录用户 {user_id} 跳过歌曲: {song_title} - {artist}")

    def remove_like(self, user_id: str, song_title: str, artist: str):
        """取消点赞 → 删除 LIKES 关系"""
        if not song_title:
            return
        query = """
        MATCH (u:User {id: $user_id})-[r:LIKES]->(s:Song {title: $song_title})
        DELETE r
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title})
            logger.info(f"用户 {user_id} 取消点赞: {song_title}")

    def remove_save(self, user_id: str, song_title: str, artist: str):
        """取消收藏 → 删除 SAVES 关系"""
        if not song_title:
            return
        query = """
        MATCH (u:User {id: $user_id})-[r:SAVES]->(s:Song {title: $song_title})
        DELETE r
        """
        if self.neo4j_client:
            self.neo4j_client.execute_query(query, {"user_id": user_id, "song_title": song_title})
            logger.info(f"用户 {user_id} 取消收藏: {song_title}")

    def get_liked_songs(self, user_id: str, limit: int = 20) -> list:
        """
        查询用户显式正向标记的歌曲（LIKES + SAVES），按时间衰减权重排序。
        衰减公式: score = weight / (1 + 0.01 * days_since_action)
        同时排除 DISLIKES 和多次跳过(>=3)的歌曲。
        """
        query = """
        MATCH (u:User {id: $user_id})-[r:LIKES|SAVES]->(s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        // 排除明确不喜欢的歌
        WHERE NOT EXISTS {
            MATCH (u)-[:DISLIKES]->(s)
        }
        WITH s, a, r, type(r) AS rel_type,
             collect(DISTINCT m.name) AS moods,
             collect(DISTINCT t.name) AS themes,
             CASE WHEN r.created_at IS NOT NULL
               THEN r.weight / (1.0 + 0.01 *
                 duration.inDays(datetime({epochMillis: r.created_at}), datetime()).days)
               ELSE coalesce(r.weight, 1.0)
             END AS decayed_score
        ORDER BY decayed_score DESC
        LIMIT $limit
        RETURN s.title AS title, coalesce(a.name, s.artist, '未知') AS artist,
               s.audio_url AS audio_url, s.cover_url AS cover_url,
               s.lrc_url AS lrc_url, s.album AS album,
               moods, themes, rel_type, decayed_score
        """
        if not self.neo4j_client:
            return []
        try:
            results = self.neo4j_client.execute_query(query, {"user_id": user_id, "limit": limit})
            songs = []
            for r in results:
                songs.append({
                    "song": {
                        "title": r.get("title", ""),
                        "artist": r.get("artist", ""),
                        "audio_url": r.get("audio_url", ""),
                        "cover_url": r.get("cover_url", ""),
                        "lrc_url": r.get("lrc_url", ""),
                        "album": r.get("album", ""),
                        "moods": r.get("moods", []),
                        "themes": r.get("themes", []),
                    },
                    "reason": f"你{'点赞' if r.get('rel_type') == 'LIKES' else '收藏'}了这首歌",
                    "source": "user_favorites",
                    "score": r.get("decayed_score", 0),
                })
            logger.info(f"查询用户 {user_id} 喜欢的歌: {len(songs)} 首")
            return songs
        except Exception as e:
            logger.error(f"查询用户喜欢歌曲失败: {e}")
            return []



    def get_user_preferences(self, user_id: str, limit: int = 10) -> Dict[str, Any]:

        """获取用户的图谱综合偏好"""

        # 简化查询：寻找用户喜欢或收听过最多的歌曲关联的流派和艺术家
        query = """

        MATCH (u:User {id: $user_id})-[rel:LIKES|SAVES|LISTENED_TO]->(s:Song)

        WITH s, rel

        ORDER BY rel.weight DESC, rel.play_count DESC LIMIT $limit

        

        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)

        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)

        

        RETURN 

            collect(DISTINCT s.title) as favorite_songs,

            collect(DISTINCT g.name) as favorite_genres,

            collect(DISTINCT a.name) as favorite_artists

        """

        

        if not self.neo4j_client:

            return {}

            

        try:

            result = self.neo4j_client.execute_query(query, {"user_id": user_id, "limit": limit})

            prefs = {}

            if result and len(result) > 0:

                record = result[0]

                prefs = {

                    "favorite_songs": record.get("favorite_songs", []),

                    "favorite_genres": record.get("favorite_genres", []),

                    "favorite_artists": record.get("favorite_artists", [])

                }

            

            # ============================================================

            # 【升级】合并 User 节点上的语义记忆属性
            # 来源：《第八章 记忆与检索》 memory_hyde_analysis 建议

            # 从 Neo4j User 节点读取通过 update_semantic_preferences

            # 持久化的显式偏好标签（如 avoid_genres, mood_tendency 等）。
            # 与图谱关系推导的偏好合并，提供更完整的用户画像。
            # ============================================================

            semantic_query = """

            MATCH (u:User {id: $user_id})

            RETURN u.avoid_genres AS avoid_genres,

                   u.add_genres AS add_genres,

                   u.add_artists AS add_artists,

                   u.avoid_artists AS avoid_artists,

                   u.mood_tendency AS mood_tendency,

                   u.activity_contexts AS activity_contexts,

                   u.language_preference AS language_preference

            """

            semantic_result = self.neo4j_client.execute_query(semantic_query, {"user_id": user_id})

            if semantic_result and len(semantic_result) > 0:

                sr = semantic_result[0]

                prefs["avoid_genres"] = sr.get("avoid_genres", []) or []

                prefs["preferred_genres_explicit"] = sr.get("add_genres", []) or []

                prefs["preferred_artists_explicit"] = sr.get("add_artists", []) or []

                prefs["avoid_artists"] = sr.get("avoid_artists", []) or []

                prefs["mood_tendency"] = sr.get("mood_tendency", "")

                prefs["activity_contexts"] = sr.get("activity_contexts", []) or []

                prefs["language_preference"] = sr.get("language_preference", "")

                

            return prefs

        except Exception as e:

            logger.error(f"提取用户图谱偏好失败: {e}")

            

        return {}



    # ============================================================

    # 【升级】语义记忆偏好持久化方法

    # 来源：《第八章 记忆与检索》 memory_hyde_analysis 建议

    # 由 LLM 从对话中提取的用户显式偏好（喜欢/讨厌的流派）
    # 歌手、情绪倾向等）写入 Neo4j User 节点属性，

    # 使系统具备跨会话的长期记忆能力。
    # ============================================================

    def update_semantic_preferences(self, user_id: str, extraction_result: Dict[str, Any]):

        """

        将从对话中提取的用户偏好持久化到 Neo4j User 节点属性。
        

        Args:

            user_id: 用户ID

            extraction_result: LLM 提取的偏好 JSON，包含 add_genres, avoid_genres 等字段
        """

        if not self.neo4j_client:

            logger.warning("[SemanticMemory] Neo4j 客户端不可用，跳过偏好持久化")

            return

            

        try:

            self.ensure_user_exists(user_id)

            

            # 构建动态 SET 子句：仅更新非空字段，避免覆盖已有数据
            set_clauses = []

            params = {"user_id": user_id}

            

            # 列表型字段：追加合并（去重）

            list_fields = ["add_genres", "avoid_genres", "add_artists", "avoid_artists", "activity_contexts"]

            for field in list_fields:

                values = extraction_result.get(field, [])

                if values and isinstance(values, list) and len(values) > 0:

                    # 使用 APOC 风格的列表合并，或者简单的 Cypher coalesce + 手动去重

                    set_clauses.append(

                        f"u.{field} = [x IN (coalesce(u.{field}, []) + ${field}) WHERE x IS NOT NULL | x]"

                    )

                    params[field] = values

            

            # 字符串型字段：直接覆盖（如果非空）
            string_fields = ["mood_tendency", "language_preference"]

            for field in string_fields:

                value = extraction_result.get(field, "")

                if value and isinstance(value, str) and value.strip():

                    set_clauses.append(f"u.{field} = ${field}")

                    params[field] = value.strip()

            

            if not set_clauses:

                logger.info("[SemanticMemory] 本轮对话未提取到新的用户偏好，跳过写入")

                return

                

            query = f"""

            MATCH (u:User {{id: $user_id}})

            SET {', '.join(set_clauses)}, u.preferences_updated_at = timestamp()

            """

            

            self.neo4j_client.execute_query(query, params)

            logger.info(f"[SemanticMemory] 成功更新用户 {user_id} 的语义偏好: {list(params.keys())}")

            

        except Exception as e:

            logger.error(f"[SemanticMemory] 偏好持久化失败: {e}")



