"""
测试 MergeAndDedup 平等合并去重逻辑

验证检索管线 Step 2 的核心行为：
- 两路结果正确合并
- 双引擎命中标记 (cross-engine hit)
- 去重逻辑 (normalize_key based dedup)

注意：直接复制核心纯函数进行测试，避免导入 hybrid_retrieval
（它会拉起 langgraph/langchain 整个依赖链）。
"""
import pytest
import json
import re
import unicodedata
from typing import List, Dict


# ---- 复制自 retrieval/hybrid_retrieval.py 的纯函数 ----

def _normalize_key(title: str, artist: str) -> str:
    """生成标准化的去重 key"""
    def _clean(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.lower().strip()
        s = re.sub(r"[,，、/\\\\\\s()（）【】\[\]]+", "", s)
        return s
    return f"{_clean(title)}_{_clean(artist)}"


BASELINE_SIMILARITY_SCORE = 0.85


def _parse_engine_results(res_str: str, engine_name: str) -> List[dict]:
    """将引擎原始 JSON 字符串解析为标准化的歌曲列表"""
    if not res_str:
        return []
    try:
        items = json.loads(res_str)
    except json.JSONDecodeError:
        return []

    results = []
    seen_keys = set()
    for rank, item in enumerate(items):
        if "error" in item:
            continue
        title = item.get("title", "未知标题")
        artist = item.get("artist", "未知艺术家")
        genre = item.get("genre", "")
        if genre == "Unknown":
            genre = ""
        key = _normalize_key(title, artist)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        raw_distance = item.get("distance", None)
        raw_similarity = item.get("similarity_score", None)
        if raw_distance is not None:
            raw_score = 1.0 / (1.0 + float(raw_distance))
        elif raw_similarity is not None:
            raw_score = float(raw_similarity)
        else:
            raw_score = BASELINE_SIMILARITY_SCORE
        results.append({
            "key": key, "rank": rank, "raw_score": raw_score,
            "engine": engine_name,
            "song": {"title": title, "artist": artist, "album": item.get("album", "未知"),
                     "genre": genre, "preview_url": None, "cover_url": None, "lrc_url": None},
        })
    return results


def _merge_and_dedup(graph_items: List[dict], vector_items: List[dict]) -> List[dict]:
    """平等合并两路检索结果"""
    song_data: Dict[str, dict] = {}
    key_engines: Dict[str, List[str]] = {}

    for item in graph_items:
        key = item["key"]
        if key not in song_data:
            song_data[key] = item
            key_engines[key] = []
        key_engines[key].append("知识图谱(GraphRAG)")

    for item in vector_items:
        key = item["key"]
        if key not in song_data:
            song_data[key] = item
            key_engines[key] = []
        if "语义向量(Neo4j Vector)" not in key_engines.get(key, []):
            key_engines.setdefault(key, []).append("语义向量(Neo4j Vector)")

    merged = []
    for key, item in song_data.items():
        engines = key_engines.get(key, [])
        both_hit = len(engines) > 1
        reason = "引擎检索来源: " + " + ".join(engines)
        if both_hit:
            reason += " 🔥双引擎交叉命中"
        merged.append({
            "song": item["song"], "reason": reason,
            "similarity_score": item["raw_score"], "_both_engines": both_hit,
        })
    merged.sort(key=lambda x: x["similarity_score"], reverse=True)
    return merged


# ---- 辅助函数 ----

def _make_item(title: str, artist: str, engine: str, score: float = 0.8) -> dict:
    """构造标准化的引擎结果项"""
    key = _normalize_key(title, artist)
    return {
        "key": key, "rank": 0, "raw_score": score, "engine": engine,
        "song": {"title": title, "artist": artist, "album": "Test Album",
                 "genre": "", "preview_url": None, "cover_url": None, "lrc_url": None},
    }


# ---- 测试 ----

class TestMergeAndDedup:
    """测试平等合并去重"""

    def test_graph_only(self):
        graph = [
            _make_item("青花瓷", "周杰伦", "知识图谱"),
            _make_item("稻香", "周杰伦", "知识图谱"),
        ]
        merged = _merge_and_dedup(graph, [])
        assert len(merged) == 2

    def test_vector_only(self):
        vector = [_make_item("夜曲", "周杰伦", "语义向量")]
        merged = _merge_and_dedup([], vector)
        assert len(merged) == 1

    def test_cross_engine_hit(self):
        """同一首歌在两个引擎都命中，应标记为双引擎交叉命中"""
        graph = [_make_item("青花瓷", "周杰伦", "知识图谱")]
        vector = [_make_item("青花瓷", "周杰伦", "语义向量")]
        merged = _merge_and_dedup(graph, vector)
        assert len(merged) == 1
        assert merged[0]["_both_engines"] is True

    def test_different_songs_preserved(self):
        graph = [_make_item("青花瓷", "周杰伦", "知识图谱")]
        vector = [_make_item("夜曲", "周杰伦", "语义向量")]
        merged = _merge_and_dedup(graph, vector)
        assert len(merged) == 2

    def test_sorted_by_score(self):
        graph = [_make_item("A", "Artist1", "知识图谱", score=0.5)]
        vector = [_make_item("B", "Artist2", "语义向量", score=0.9)]
        merged = _merge_and_dedup(graph, vector)
        assert merged[0]["song"]["title"] == "B"
        assert merged[1]["song"]["title"] == "A"

    def test_empty_inputs(self):
        merged = _merge_and_dedup([], [])
        assert len(merged) == 0


class TestParseEngineResults:
    """测试引擎原始 JSON 解析"""

    def test_valid_json(self):
        data = json.dumps([
            {"title": "青花瓷", "artist": "周杰伦", "album": "我很忙"},
            {"title": "稻香", "artist": "周杰伦", "album": "魔杰座"},
        ])
        results = _parse_engine_results(data, "test")
        assert len(results) == 2
        assert results[0]["song"]["title"] == "青花瓷"

    def test_empty_string(self):
        results = _parse_engine_results("", "test")
        assert results == []

    def test_invalid_json(self):
        results = _parse_engine_results("not json}", "test")
        assert results == []

    def test_dedup_within_engine(self):
        data = json.dumps([
            {"title": "青花瓷", "artist": "周杰伦"},
            {"title": "青花瓷", "artist": "周杰伦"},
        ])
        results = _parse_engine_results(data, "test")
        assert len(results) == 1
