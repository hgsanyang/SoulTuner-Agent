"""
测试 _normalize_key 去重逻辑

验证：全角/半角、标点、空格、大小写差异 → 同一首歌应生成相同的 key。
这是检索管线去重的基础，不能出错。

注意：直接复制 _normalize_key 的纯函数逻辑来测试，
避免导入 hybrid_retrieval（它会拉起整个 langgraph 依赖链）。
"""
import pytest
import re
import unicodedata


def _normalize_key(title: str, artist: str) -> str:
    """
    复制自 retrieval/hybrid_retrieval.py MusicHybridRetrieval._normalize_key
    生成标准化的去重 key，消除全角/半角、标点、空格差异。
    """
    def _clean(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)  # 全角→半角
        s = s.lower().strip()
        s = re.sub(r'[,，、/\\\s()（）【】\[\]]+', '', s)  # 去掉标点和空格
        return s
    return f"{_clean(title)}_{_clean(artist)}"


class TestNormalizeKey:
    """测试去重 key 标准化逻辑"""

    def test_identical_input(self):
        """完全相同的输入应产生相同的 key"""
        key1 = _normalize_key("青花瓷", "周杰伦")
        key2 = _normalize_key("青花瓷", "周杰伦")
        assert key1 == key2

    def test_fullwidth_halfwidth(self):
        """全角与半角字符应被视为同一首歌（NFKC 标准化）"""
        key_full = _normalize_key("青花瓷（Live）", "周杰伦")
        key_half = _normalize_key("青花瓷(Live)", "周杰伦")
        assert key_full == key_half

    def test_case_insensitive(self):
        """大小写不敏感"""
        key_lower = _normalize_key("hello", "Adele")
        key_upper = _normalize_key("Hello", "ADELE")
        assert key_lower == key_upper

    def test_whitespace_ignored(self):
        """空格应被忽略"""
        key1 = _normalize_key("Dont Stop", "Fleetwood Mac")
        key2 = _normalize_key("DontStop", "FleetwoodMac")
        assert key1 == key2

    def test_punctuation_ignored(self):
        """常见标点应被忽略"""
        key1 = _normalize_key("稻香", "周杰伦")
        key2 = _normalize_key("稻香，", "周杰伦、")
        assert key1 == key2

    def test_different_songs_different_keys(self):
        """不同歌曲应产生不同的 key"""
        key1 = _normalize_key("青花瓷", "周杰伦")
        key2 = _normalize_key("稻香", "周杰伦")
        assert key1 != key2

    def test_different_artists_different_keys(self):
        """同名歌曲不同歌手应产生不同的 key"""
        key1 = _normalize_key("夜曲", "周杰伦")
        key2 = _normalize_key("夜曲", "肖邦")
        assert key1 != key2

    def test_empty_input(self):
        """空输入不应报错"""
        key = _normalize_key("", "")
        assert isinstance(key, str)
        assert "_" in key  # 格式是 title_artist
