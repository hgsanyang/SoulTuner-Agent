import logging
import re
import urllib.parse
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from config.settings import settings

# 懒加载 neo4j 客户端：如果 neo4j 包未安装，允许系统降级运行（图谱检索返回空，其他检索正常工作）

_neo4j_available = True

try:

    from retrieval.neo4j_client import get_neo4j_client

except Exception as _neo4j_import_err:

    _neo4j_available = False
    logger.warning(f"Neo4j 客户端导入失败（neo4j 包可能未安装），图谱检索将被跳过: {_neo4j_import_err}")
    def get_neo4j_client():
        return None

# ── 倒排索引：title（文件名）→ track_id，用于为 GraphRAG 结果补充 preview_url ──

# Neo4j 里存储 song.name = MTG raw.meta.tsv 中的文件名（如 "1317838.low.mp3"）
# 这里建立 filename → track_id 的映射，再由 track_id → 真实信息

_TITLE_TO_TRACKID_CACHE: Optional[Dict[str, str]] = None

def _get_title_to_trackid() -> Dict[str, str]:

    """懒加载：filename → track_id 倒排索引"""
    global _TITLE_TO_TRACKID_CACHE
    if _TITLE_TO_TRACKID_CACHE is not None:
        return _TITLE_TO_TRACKID_CACHE
    _TITLE_TO_TRACKID_CACHE = {}
    # 【V2 升级】旧版 vector_search 依赖已移除，索引无需预建
    return _TITLE_TO_TRACKID_CACHE

def _build_preview_url(raw_title: str) -> Optional[str]:

    """
    给定 Neo4j 中存储的原始文件名（如 '1317838.low.mp3'），
    构建与向量检索相同格式的本地试听 URL。
    """
    if not raw_title:
        return None
    try:
        safe_title = raw_title.replace('\\', '/')
        # 按 MTG Jamendo 子文件夹规则：按 ID 后两位分目录
        if '/' not in safe_title:
            match = re.search(r'(\d+)', safe_title)
            if match:
                num_str = match.group(1)
                folder = num_str[-2:].zfill(2)
                safe_title = f"{folder}/{safe_title}"
        return f"{settings.api_base_url}/static/audio/{urllib.parse.quote(safe_title)}"
    except Exception:
        return None

# ============================================================

# 流派标签映射表：中文流派 → Neo4j Theme 节点英文关键词
# 对应 Neo4j 关系：(Song)-[:HAS_THEME]->(Theme)

# ============================================================

GENRE_TAG_MAP: Dict[str, List[str]] = {

    # 电子音乐大类
    "电子": ["electronic", "electro"],
    "电子舞曲": ["electronic", "dance", "edm", "electro"],
    "舞曲": ["dance", "electronic", "edm"],
    "电舞": ["edm", "electronic", "dance"],
    "EDM": ["edm", "electronic", "dance"],
    "浩室": ["house"],
    "豪斯": ["house"],
    "House": ["house"],
    "科技舞曲": ["techno"],
    "迷幻舞曲": ["trance"],
    "鼓打贝斯": ["drum and bass", "dnb"],
    "鼓打碎拍": ["drum and bass", "breakbeat"],
    "迷幻贝斯": ["dubstep"],
    "双步舞曲": ["dubstep"],
    "环境电子": ["ambient", "electronic"],
    "氛围": ["ambient", "chillout"],
    "氛围电子": ["ambient", "electronic"],
    "Ambient": ["ambient"],
    "chillout": ["chillout", "ambient"],
    "放松音乐": ["chillout", "ambient", "relaxing"],
    "合成器流行": ["synthpop"],
    "电子流行": ["electropop", "synthpop"],
    "赛博朋克": ["industrial", "electronic", "synthpop"],
    "工业": ["industrial"],
    "工业音乐": ["industrial", "electronic"],
    "科幻电子": ["industrial", "electronic"],
    "未来贝斯": ["future bass", "electronic"],
    "迷幻": ["psychedelic", "psychedelic rock"],
    "迷幻摇滚": ["psychedelic rock", "psychedelic"],
    "电音": ["electronic", "electro", "edm"],
    "Techno": ["techno"],
    "陷阱": ["trap", "electronic"],
    "Trap": ["trap"],
    "Trip-hop": ["trip-hop", "triphop"],
    "旅途嘻哈": ["trip-hop", "triphop"],
    # 摇滚大类
    "摇滚": ["rock"],
    "国摇": ["rock"],
    "华语摇滚": ["rock"],
    "中国摇滚": ["rock"],
    "日摇": ["rock"],
    "经典摇滚": ["classic rock", "rock"],
    "替代摇滚": ["alternative rock", "alternative"],
    "另类": ["alternative", "alternative rock"],
    "另类摇滚": ["alternative rock", "alternative"],
    "独立摇滚": ["indie rock", "indie"],
    "独立": ["indie", "indie rock"],
    "硬摇滚": ["hard rock"],
    "重摇滚": ["hard rock", "heavy metal"],
    "朋克摇滚": ["punk rock", "punk"],
    "朋克": ["punk", "punk rock"],
    "流行朋克": ["pop punk"],
    "后朋克": ["post-punk"],
    "车库摇滚": ["garage rock"],
    "叙事摇滚": ["folk rock", "folk"],
    "布鲁斯摇滚": ["blues rock", "blues"],
    "放克摇滚": ["funk rock", "funk"],
    "说唱摇滚": ["rap rock", "rap"],
    "渐进摇滚": ["progressive rock"],
    "前卫摇滚": ["progressive rock"],
    "鞋声": ["shoegaze"],
    "噪音摇滚": ["noise rock", "experimental"],
    "后摇": ["post-rock"],
    "数学摇滚": ["math rock"],
    "情绪": ["emo", "emotional"],
    "情绪硬核": ["emo", "hardcore"],
    "油渍摇滚": ["grunge"],
    "垃圾摇滚": ["grunge"],
    "英伦摇滚": ["britpop", "indie rock"],
    "新浪潮": ["new wave"],
    "哥特": ["gothic", "darkwave"],
    "哥特摇滚": ["gothic", "rock"],
    # 金属大类
    "金属": ["metal"],
    "重金属": ["heavy metal", "metal"],
    "死亡金属": ["death metal", "metal"],
    "黑金属": ["black metal", "metal"],
    "激流金属": ["thrash metal", "metal"],
    "旋律死亡金属": ["melodic death metal", "metal"],
    "力量金属": ["power metal", "metal"],
    "哥特金属": ["gothic metal", "metal"],
    "前卫金属": ["progressive metal", "metal"],
    "核心": ["metalcore", "hardcore"],
    "金属核": ["metalcore"],
    "死亡核": ["deathcore"],
    "硬核": ["hardcore"],
    "后硬核": ["post-hardcore"],
    "嘶吼": ["screamo", "post-hardcore"],
    # 嘻哈/说唱
    "嘻哈": ["hip hop", "hip-hop"],
    "说唱": ["rap", "hip hop"],
    "힙합": ["hip hop"],
    "饶舌": ["rap", "hip hop"],
    "陷阱嘻哈": ["trap", "hip hop"],
    "老派嘻哈": ["old school hip hop", "hip hop"],
    "西海岸说唱": ["west coast rap", "hip hop"],
    "东海岸说唱": ["east coast rap", "hip hop"],
    "地下嘻哈": ["underground hip hop", "hip hop"],
    # 蓝调/灵魂
    "蓝调": ["blues"],
    "布鲁斯": ["blues"],
    "灵魂": ["soul"],
    "灵乐": ["soul", "gospel"],
    "节奏布鲁斯": ["rnb", "r&b", "soul"],
    "RnB": ["rnb", "soul"],
    "R&B": ["rnb", "r&b"],
    "放克": ["funk"],
    "放克灵魂": ["funk", "soul"],
    "爵士蓝调": ["jazz blues", "blues", "jazz"],
    "蓝调摇滚": ["blues rock", "blues", "rock"],
    "传统蓝调": ["blues"],
    "三角洲蓝调": ["blues"],
    # 爵士
    "爵士": ["jazz"],
    "爵士乐": ["jazz"],
    "酸爵士": ["acidjazz", "jazz"],
    "融合爵士": ["jazz fusion", "jazz"],
    "现代爵士": ["jazz", "modern jazz"],
    "自由爵士": ["free jazz", "jazz"],
    "摇摆乐": ["swing", "jazz"],
    "比波普": ["jazz", "bebop"],
    "冷爵士": ["cool jazz", "jazz"],
    # 流行
    "流行": ["pop"],
    "流行音乐": ["pop"],
    "流行摇滚": ["pop rock"],
    "独立流行": ["indie pop"],
    "梦幻流行": ["dream pop"],
    "电音流行": ["electropop", "synthpop"],
    "慵懒氛围": ["chillwave", "ambient"],
    "氛围流行": ["chillwave", "dream pop"],
    # 民谣
    "民谣": ["folk", "acoustic"],
    "民俗音乐": ["folk"],
    "乡村民谣": ["country folk", "folk", "country"],
    "美式民谣": ["americana", "folk"],
    "创作者": ["singersongwriter", "folk"],
    "抒情创作者": ["singersongwriter", "folk"],
    # 乡村
    "乡村": ["country"],
    "乡村音乐": ["country"],
    "Alternative Country": ["alt-country", "country"],
    "另类乡村": ["alt-country", "country"],
    "蓝草": ["bluegrass", "folk"],
    # 古典/音景
    "古典": ["classical"],
    "古典音乐": ["classical"],
    "交响": ["classical", "symphonic"],
    "弦乐": ["classical", "orchestral"],
    "器乐": ["instrumental", "classical"],
    "新古典": ["neoclassical", "classical"],
    "极简主义": ["minimal", "classical"],
    "环境音乐": ["ambient"],
    "冥想音乐": ["ambient", "chillout"],
    "原声": ["acoustic"],
    "原声吉他": ["acoustic", "singersongwriter"],
    # 世界音乐
    "世界音乐": ["world", "worldfusion"],
    "拉丁": ["latin"],
    "波萨诺瓦": ["bossa nova", "latin", "jazz"],
    "雷鬼": ["reggae"],
    "雷鬼顿": ["reggaeton", "latin"],
    "达布": ["dub", "reggae"],
    "斯卡": ["ska", "reggae"],
    "非洲音乐": ["afro", "world"],
    "凯尔特": ["celtic", "folk"],
    "蓝草音乐": ["bluegrass"],
    "弗拉明戈": ["flamenco", "latin"],
    "桑巴": ["samba", "latin", "world"],
    # 宗教/圣乐
    "福音乐": ["gospel"],
    "赞美诗": ["worship", "gospel"],
    "崇拜音乐": ["worship"],
    "基督": ["christian"],
    # 迪斯科/舞池
    "迪斯科": ["disco"],
    "迪士科": ["disco"],
    "霓虹舞": ["new wave", "disco"],
    "舞曲流行": ["dance pop", "dance", "pop"],
    # 特殊/杂类
    "原声带": ["soundtrack"],
    "配乐": ["soundtrack"],
    "电影音乐": ["soundtrack", "classical"],
    "游戏音乐": ["soundtrack", "electronic"],
    "低保真": ["lo-fi"],
    "Lo-fi": ["lo-fi"],
    "噪音": ["noise", "experimental"],
    "实验": ["experimental"],
    "实验音乐": ["experimental"],
    "后现代": ["experimental", "avant-garde"],
    "新世纪": ["new age", "ambient"],
    "轻音乐": ["easy listening", "ambient"],

}

# ============================================================

# 场景标签映射表：中文活动场景 → Neo4j Scenario 节点英文关键词
# 对应 Neo4j 关系：(Song)-[:FITS_SCENARIO]->(Scenario)

# ============================================================

SCENARIO_TAG_MAP: Dict[str, List[str]] = {

    "运动": ["workout", "energetic", "sport"],
    "健身": ["workout", "energetic", "sport"],
    "跑步": ["workout", "energetic", "sport"],
    "学习": ["study", "peaceful"],
    "工作": ["work", "study"],
    "开车": ["driving", "energetic"],
    "睡觉": ["sleep", "relaxing"],
    "睡前": ["sleep", "relaxing", "peaceful"],
    "派对": ["party", "energetic", "happy"],
    "聚会": ["party", "happy"],
    "旅行": ["travel", "journey"],
    "通勤": ["commute", "morning"],
    "做饭": ["cooking", "relaxing"],
    "冥想": ["relaxing", "peaceful", "healing", "relaxation"],
    "下雨天": ["rainy day", "melancholy"],
    # ── MTG 新增场景 ──
    "看电影": ["film", "movie", "soundtrack"],
    "打游戏": ["game", "gaming"],

}

# ============================================================

# 情绪标签映射表：中文情绪 → Neo4j Mood 节点英文关键词
# 对应 Neo4j 关系：(Song)-[:HAS_MOOD]->(Mood)

# ============================================================

MOOD_TAG_MAP: Dict[str, List[str]] = {

    "开心": ["happy", "energetic"],
    "快乐": ["happy", "energetic"],
    "悲伤": ["melancholy", "lonely", "sad"],
    "伤感": ["melancholy", "lonely", "sad"],
    "忧郁": ["melancholy", "dark"],
    "愤怒": ["angry", "raw"],
    "暴躁": ["angry", "raw", "energetic"],
    "放松": ["relaxing", "peaceful"],
    "平静": ["peaceful", "relaxing"],
    "浪漫": ["romantic", "dreamy"],
    "怀旧": ["nostalgic", "melancholy"],
    "治愈": ["healing", "hopeful"],
    "孤独": ["lonely", "melancholy"],
    "热血": ["energetic", "angry", "raw"],
    "激动": ["energetic", "happy"],
    "梦幻": ["dreamy", "peaceful"],
    # 新增缺失的情绪词
    "深情": ["romantic", "emotional", "melancholy"],
    "温柔": ["gentle", "romantic", "peaceful"],
    "感动": ["emotional", "hopeful"],
    "温暖": ["warm", "hopeful", "gentle"],
    "深沉": ["dark", "melancholy", "emotional"],
    "激情": ["energetic", "passionate", "raw"],
    "壮阔": ["epic", "energetic", "hopeful"],
    "柔情": ["romantic", "gentle", "emotional"],
    "忧伤": ["melancholy", "lonely", "emotional"],
    "惆怅": ["melancholy", "nostalgic", "lonely"],
    "燃": ["energetic", "raw", "angry"],
    "带感": ["energetic", "happy", "raw"],
    "抒情": ["emotional", "romantic", "melancholy"],
    "沉醉": ["dreamy", "romantic", "emotional"],
    "恐怖": ["eerie", "dark", "horror"],

}

# ============================================================

# 主题标签映射表：中文主题 → Neo4j Theme 节点英文关键词
# 对应 Neo4j 关系：(Song)-[:HAS_THEME]->(Theme)

# ============================================================

THEME_TAG_MAP: Dict[str, List[str]] = {

    "爱情": ["love", "romantic", "heartbreak"],
    "友情": ["friendship", "happy"],
    "自由": ["freedom", "journey"],
    "青春": ["youth", "growth"],
    "成长": ["growth", "youth"],
    "夜晚": ["night", "late night"],
    "深夜": ["late night", "night"],
    "早晨": ["morning", "hopeful"],
    "自然": ["nature", "peaceful"],
    "城市": ["urban", "night"],
    "失恋": ["heartbreak", "melancholy"],
    # ── MTG 新增主题 ──
    "电影": ["film", "movie", "soundtrack"],
    "游戏": ["game", "gaming"],
    "恐怖": ["horror", "eerie", "dark"],
    "史诗": ["epic", "adventure"],
    "流浪": ["travel", "journey"],

}

# ============================================================

# 合并字典（供确定性后处理扫描使用）
# ============================================================

ALL_TAG_MAPS = {

    "genre": GENRE_TAG_MAP,
    "scenario": SCENARIO_TAG_MAP,
    "mood": MOOD_TAG_MAP,
    "theme": THEME_TAG_MAP,

}

def _expand_tag(tag_str: Optional[str], tag_map: Dict[str, List[str]]) -> List[str]:

    """通用标签扩展：精确→大小写不敏感→部分包含→原样返回"""
    if not tag_str:
        return []
    tag_str = tag_str.strip()
    if tag_str in tag_map:
        return tag_map[tag_str]
    tag_lower = tag_str.lower()
    for key, aliases in tag_map.items():
        if key.lower() == tag_lower:
            return aliases
    results = []
    for key, aliases in tag_map.items():
        if key in tag_str or tag_str in key:
            results.extend(aliases)
    if results:
        return list(dict.fromkeys(results))
    return [tag_str.lower()]

# ============================================================

# 语言别名映射表：用户中文简称 → Neo4j Language 节点标准名
# ============================================================

LANGUAGE_ALIAS_MAP: Dict[str, str] = {

    # 中文/国语
    "中文": "Chinese",
    "国语": "Chinese",
    "普通话": "Chinese",
    "汉语": "Chinese",
    "中文歌": "Chinese",
    "国语歌": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    # 复合概念词隐含语言
    "国摇": "Chinese",
    "华语摇滚": "Chinese",
    "中国摇滚": "Chinese",
    "华语": "Chinese",
    "华语歌": "Chinese",
    "国风": "Chinese",
    "古风": "Chinese",
    # 粤语
    "粤语": "Cantonese",
    "广东歌": "Cantonese",
    "粤语歌": "Cantonese",
    "港乐": "Cantonese",
    "cantonese": "Cantonese",
    # 英语
    "英文": "English",
    "英语": "English",
    "英文歌": "English",
    "英语歌": "English",
    "欧美歌": "English",
    "english": "English",
    # 日语
    "日语": "Japanese",
    "日文": "Japanese",
    "日语歌": "Japanese",
    "日文歌": "Japanese",
    "日摇": "Japanese",
    "j-pop": "Japanese",
    "japanese": "Japanese",
    # 韩语
    "韩语": "Korean",
    "韩文": "Korean",
    "韩语歌": "Korean",
    "韩文歌": "Korean",
    "韩流": "Korean",
    "k-pop": "Korean",
    "korean": "Korean",
    # 器乐
    "器乐": "Instrumental",
    "纯音乐": "Instrumental",
    "instrumental": "Instrumental",

}

# ============================================================

# 地区别名映射表：用户中文表达 → Neo4j Region 节点标准名
# ============================================================

REGION_ALIAS_MAP: Dict[str, str] = {

    # 内地
    "内地": "Mainland China",
    "内地歌": "Mainland China",
    "国内": "Mainland China",
    "国内歌": "Mainland China",
    "内地音乐": "Mainland China",
    # 台湾
    "台湾": "Taiwan",
    "台香": "Taiwan",
    "台湾歌": "Taiwan",
    "台语": "Taiwan",
    # 香港
    "香港": "Hong Kong",
    "港乐": "Hong Kong",
    "香港歌": "Hong Kong",
    # 日本
    "日本": "Japan",
    "日语歌": "Japan",
    "小语种日语": "Japan",
    # 韩国
    "韩国": "Korea",
    "韩国歌": "Korea",
    "kpop": "Korea",
    "k-pop": "Korea",
    # 欧美
    "欧美": "Western",
    "欧美歌": "Western",
    "欧美音乐": "Western",
    "西方": "Western",
    "西方歌": "Western",
    "western": "Western",

}

def _expand_genre_to_english(genre_str: Optional[str]) -> List[str]:

    """将中文流派字符串通过 GENRE_TAG_MAP 展开为英文关键词列表。"""
    return _expand_tag(genre_str, GENRE_TAG_MAP)

def _expand_scenario_to_english(scenario_str: Optional[str]) -> List[str]:

    """将中文场景字符串通过 SCENARIO_TAG_MAP 展开为英文关键词列表。"""
    return _expand_tag(scenario_str, SCENARIO_TAG_MAP)

def _expand_mood_to_english(mood_str: Optional[str]) -> List[str]:

    """将中文情绪字符串通过 MOOD_TAG_MAP 展开为英文关键词列表。"""
    return _expand_tag(mood_str, MOOD_TAG_MAP)

def _expand_theme_to_english(theme_str: Optional[str]) -> List[str]:

    """将中文主题字符串通过 THEME_TAG_MAP 展开为英文关键词列表。"""
    return _expand_tag(theme_str, THEME_TAG_MAP)

@tool("graphrag_search")

def graphrag_search(query: str, limit: int = 5) -> str:

    """
    Music Knowledge Graph Search Tool (Neo4j GraphRAG).
    Use this tool when the user asks complex relational questions about music, such as:
    - "Who collaborated with Michael Jackson on 'Thriller'?"
    - "Which artists belong to the same sub-genre and record label as Taylor Swift?"
    - "Find the shortest connection between The Beatles and modern K-pop artists."
    Args:
        query: The natural language question or specific entity to search for.
        limit: Maximum number of paths or relationships to return.
    """
    logger.info(f"Executing GraphRAG search for: {query}")
    client = get_neo4j_client()
    if not client.driver:
        return "Warning: Neo4j database is currently disconnected. Cannot perform GraphRAG search."
    try:
        import json
        # ── 解析传入的 query（JSON 格式或纯文本）──
        try:
            intent = json.loads(query)
            tags = intent.get("tags", [])
            genre = intent.get("genre")
            scenario = intent.get("scenario")
            mood = intent.get("mood")
            language_raw = intent.get("language", None)
            region_raw = intent.get("region", None)
            # 异常值清理
            if tags in ["[]", "null", "None", "", None]:
                tags = []
            elif isinstance(tags, str):
                try:
                    import ast
                    parsed_tags = ast.literal_eval(tags)
                    tags = parsed_tags if isinstance(parsed_tags, list) else [tags]
                except (ValueError, SyntaxError):
                    tags = [tags]
            for field_name in ["genre", "scenario", "mood"]:
                val = locals().get(field_name)
                if val in ["[]", "null", "None", "", [], None]:
                    locals()[field_name] = None
            # 重新赋值（locals() 赋值在函数作用域不可靠）
            if genre in ["[]", "null", "None", "", [], None]:
                genre = None
            if scenario in ["[]", "null", "None", "", [], None]:
                scenario = None
            if mood in ["[]", "null", "None", "", [], None]:
                mood = None
            if language_raw in ["null", "None", "", None]:
                language_raw = None
            if region_raw in ["null", "None", "", None]:
                region_raw = None
        except json.JSONDecodeError:
            # 降级：纯文本当 tag 处理
            tags = [query]
            genre = None
            scenario = None
            mood = None
            language_raw = None
            region_raw = None
        # ── 语言解析：中文别名 → 标准 Language 节点名──
        language_normalized = None
        if language_raw:
            lang_key = language_raw.strip()
            language_normalized = (
                LANGUAGE_ALIAS_MAP.get(lang_key)
                or LANGUAGE_ALIAS_MAP.get(lang_key.lower())
                or lang_key
            )
            logger.info(f"GraphRAG 语言解析: '{language_raw}' → '{language_normalized}'")
        # ── 地区解析：中文别名 → 标准 Region 节点名──
        region_normalized = None
        if region_raw:
            reg_key = region_raw.strip()
            region_normalized = (
                REGION_ALIAS_MAP.get(reg_key)
                or REGION_ALIAS_MAP.get(reg_key.lower())
                or reg_key
            )
            logger.info(f"GraphRAG 地区解析: '{region_raw}' → '{region_normalized}'")
        # ── 三维标签扩展（各自走各自的字典）──
        genre_aliases = _expand_genre_to_english(genre) if genre else []
        scenario_aliases = _expand_scenario_to_english(scenario) if scenario else []
        mood_aliases = _expand_mood_to_english(mood) if mood else []
        if genre_aliases:
            logger.info(f"GraphRAG 流派扩展: '{genre}' → {genre_aliases}")
        if scenario_aliases:
            logger.info(f"GraphRAG 场景扩展: '{scenario}' → {scenario_aliases}")
        if mood_aliases:
            logger.info(f"GraphRAG 情绪扩展: '{mood}' → {mood_aliases}")
        # ── 纯音乐特殊路由：'instrumental' 是 Language 属性而非 Theme/Mood ──
        _INSTRUMENTAL_ALIASES = {"instrumental", "器乐", "纯音乐"}
        _genre_lower = (genre or "").lower()
        _is_instrumental_query = (
            _genre_lower in _INSTRUMENTAL_ALIASES
            or any(a in _INSTRUMENTAL_ALIASES for a in genre_aliases)
        )
        if _is_instrumental_query and not language_normalized:
            language_normalized = "Instrumental"
            genre = None
            genre_aliases = []
            logger.info("GraphRAG 纯音乐查询：已自动路由至 Language=Instrumental 筛选")
        # ── tag 清洗：过滤无意义的中文长句（fallback 误传的原始查询）──
        _CN_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]')
        cleaned_tags = []
        for tag in tags:
            cn_count = len(_CN_CHAR_PATTERN.findall(tag))
            if cn_count >= 5 and tag not in GENRE_TAG_MAP and tag not in SCENARIO_TAG_MAP and tag not in MOOD_TAG_MAP:
                logger.warning(f"检测到无效中文长句 tag '{tag}'（含 {cn_count} 个汉字），已过滤")
            else:
                cleaned_tags.append(tag)
        tags = cleaned_tags
        # ============================================================
        # 构建 Cypher 查询
        # Neo4j Schema：
        #   (Song)-[:PERFORMED_BY]->(Artist)
        #   (Song)-[:HAS_THEME]->(Theme)       → genre_aliases 匹配这里
        #   (Song)-[:HAS_MOOD]->(Mood)          → mood_aliases 匹配这里
        #   (Song)-[:FITS_SCENARIO]->(Scenario) → scenario_aliases 匹配这里
        #   (Song)-[:HAS_LANGUAGE]->(Language)
        #   (Song)-[:IN_REGION]->(Region)
        # ============================================================
        cypher_query = "MATCH (s:Song)-[:PERFORMED_BY]->(a:Artist)\n"
        # ──「tag 条件」：实体名（歌手/歌名）精确匹配──
        tag_conditions = []
        if tags:
            if len(tags) == 1 and len(tags[0]) > 20:
                logger.warning(f"检测到 tags 是一个长句子 '{tags[0]}'，短路图谱检索。")
                return json.dumps([])
            for i, tag in enumerate(tags):
                tag_conditions.append(
                    f"(toLower(s.title) CONTAINS toLower($tags[{i}]) "
                    f"OR toLower(a.name) CONTAINS toLower($tags[{i}]))"
                )
            cypher_query += f"WHERE {' OR '.join(tag_conditions)}\n"
        elif not genre_aliases and not scenario_aliases and not mood_aliases and not language_normalized and not region_normalized:
            logger.warning("GraphRAG 参数全空（tags/genre/scenario/mood/language/region），避免盲捞。")
            return json.dumps([])
        # ──「流派条件」：genre_aliases → Genre 节点 + Theme 节点双路匹配 ──
        has_genre_filter = bool(genre_aliases)
        if has_genre_filter:
            genre_conds = " OR ".join([f"toLower(g.name) CONTAINS '{a}'" for a in genre_aliases])
            theme_conds = " OR ".join([f"toLower(t.name) CONTAINS '{a}'" for a in genre_aliases])
            # 双路匹配: Genre 节点或 Theme 节点，任一命中即可
            cypher_query += "OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)\n"
            cypher_query += "OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)\n"
            cypher_query += f"WITH s, a, g, t WHERE ({genre_conds}) OR ({theme_conds})\n"
        else:
            cypher_query += "OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)\n"
            cypher_query += "OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)\n"
        # ──「情绪条件」：mood_aliases → Mood 节点 ──
        has_mood_filter = bool(mood_aliases)
        if has_mood_filter:
            mood_conds = " OR ".join([f"toLower(m.name) CONTAINS '{a}'" for a in mood_aliases])
            cypher_query += "MATCH (s)-[:HAS_MOOD]->(m:Mood)\n"
            cypher_query += f"WHERE ({mood_conds})\n"
        else:
            cypher_query += "OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)\n"
        # ──「场景条件」：scenario_aliases → Scenario 节点 ──
        has_scenario_filter = bool(scenario_aliases)
        if has_scenario_filter:
            sc_conds = " OR ".join([f"toLower(sc.name) CONTAINS '{a}'" for a in scenario_aliases])
            cypher_query += "MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)\n"
            cypher_query += f"WHERE ({sc_conds})\n"
        else:
            cypher_query += "OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)\n"
        # ──「语言 + 地区条件」── 属性过滤（不是关系节点）
        prop_filters = []
        if language_normalized:
            prop_filters.append(f"toLower(s.language) = toLower('{language_normalized}')")
        if region_normalized:
            prop_filters.append(f"toLower(s.region) = toLower('{region_normalized}')")
        if prop_filters:
            filter_str = " AND ".join(prop_filters)
            cypher_query += f"WITH s, a, g, t, m, sc WHERE {filter_str}\n"
        # ── 个性化排序由 Graph Affinity (hybrid_retrieval.py) 完成 ──
        # 不在图谱查询中使用 WITH 链（会破坏 collect(DISTINCT) 聚合）
        # ── RETURN + ORDER BY ──
        if tags:
            order_by_exact = " OR ".join([f"toLower(a.name) = toLower($tags[{i}])" for i in range(len(tags))])
            order_by_contains = " OR ".join([f"toLower(a.name) CONTAINS toLower($tags[{i}])" for i in range(len(tags))])
            order_clause = f"""ORDER BY
            CASE WHEN {order_by_exact} THEN 0
                 WHEN {order_by_contains} THEN 1
                 ELSE 2 END ASC"""
        else:
            order_clause = "ORDER BY s.title ASC"
        cypher_query += f"""RETURN s.title AS track_name, s.music_id AS raw_name, s.title AS title,
        a.name AS artist,
        collect(DISTINCT t.name) AS themes,
        collect(DISTINCT g.name) AS genres,
        collect(DISTINCT m.name) AS moods,
        collect(DISTINCT sc.name) AS scenarios,
        coalesce(s.language, 'Unknown') AS language,
        coalesce(s.region, 'Unknown') AS region,
        s.album AS album,
        s.audio_url AS audio_url,
        s.cover_url AS cover_url,
        s.lrc_url AS lrc_url
        {order_clause}
        LIMIT $limit
        """
        logger.info(f"Agent 触发执行图谱精准 Cypher: {cypher_query}")
        results = client.execute_query(cypher_query, {"tags": tags, "limit": limit})
        structured_results = []
        if results:
            BASE_API_URL = settings.api_base_url
            for record in results:
                raw_name = record.get("raw_name", "") or ""
                track_name = record.get("title") or record.get("track_name", "Unknown")
                audio_url = record.get("audio_url", "") or ""
                preview_url = f"{BASE_API_URL}{audio_url}" if audio_url else None
                cover_url = record.get("cover_url", "") or ""
                cover_url = f"{BASE_API_URL}{cover_url}" if cover_url else None
                lrc_url = record.get("lrc_url", "") or ""
                lrc_url = f"{BASE_API_URL}{lrc_url}" if lrc_url else None
                # 图谱匹配置信度评分
                graph_score = 0.85
                artist_name = record.get("artist", "").lower()
                if tags:
                    for tag in tags:
                        if tag and tag.lower() == artist_name:
                            graph_score = 1.0
                            break
                        elif tag and tag.lower() in artist_name:
                            graph_score = 0.95
                # 构建 genre 展示字段
                themes_list = [x for x in (record.get("themes") or []) if x]
                genres_list = [x for x in (record.get("genres") or []) if x]
                moods_list  = [x for x in (record.get("moods")  or []) if x]
                scenarios_list = [x for x in (record.get("scenarios") or []) if x]
                # 优先使用 Genre 节点名（真实音乐流派），没有则 fallback 到 Theme
                genre_val = "/".join(genres_list[:2]) if genres_list else (themes_list[0] if themes_list else "Unknown")
                mood_val  = moods_list[0]  if moods_list  else "Unknown"
                scenario_val = scenarios_list[0] if scenarios_list else "Unknown"
                genre_display = "/".join(filter(lambda x: x != "Unknown", [genre_val, mood_val, scenario_val])) or "Unknown"
                structured_results.append({
                    "title": track_name,
                    "artist": record.get("artist", "Unknown"),
                    "album": record.get("album", "未知"),
                    "genre": genre_display,
                    "language": record.get("language", "Unknown"),
                    "region": record.get("region", "Unknown"),
                    "source": "GraphRAG",
                    "preview_url": preview_url,
                    "cover_url": cover_url,
                    "lrc_url": lrc_url,
                    "similarity_score": graph_score,
                })
        logger.info(f"GraphRAG 返回 {len(structured_results)} 条结果")
        return json.dumps(structured_results, ensure_ascii=False)
    except Exception as e:
        logger.error(f"GraphRAG Search failed: {e}")
        return json.dumps([{"error": f"Error executing graph search: {str(e)}\n{type(e).__name__}"}])
