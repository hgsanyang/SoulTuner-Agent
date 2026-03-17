# ============================================================
# 【歌词自动标签提取器】— 通过 API 调用 LLM 替代手动 Gemini 网页端
#
# 功能：
#   1. 读取 lyrics/*.lrc 歌词文件
#   2. 分批发送给 LLM API（DashScope/SiliconFlow/Google 等）
#   3. 自动解析 JSON 标签并合并写入 gemini_result.json
#
# 用法：
#   # 使用 SiliconFlow DeepSeek-V3（默认，推荐）
#   python data_pipeline/lyrics_analyzer.py
#
#   # 使用 DashScope Qwen3.5-plus
#   python data_pipeline/lyrics_analyzer.py --provider dashscope --model qwen3.5-plus
#
#   # 使用 Google Gemini
#   python data_pipeline/lyrics_analyzer.py --provider google
#
#   # 只看会处理哪些歌，不实际调用 API
#   python data_pipeline/lyrics_analyzer.py --dry-run
#
# 说明：
#   - 输出格式与 prepare_gemini_lrc_prompt.py + Gemini 手动流程完全兼容
#   - 已处理过的歌曲会自动跳过（防重复）
#   - python data_pipeline/prepare_gemini_lrc_prompt.py 仍可继续使用（手动流程保留）
# ============================================================

import os
import sys
import re
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- 目录配置（与 prepare_gemini_lrc_prompt.py 保持一致）----
LYRICS_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\lyrics"
METADATA_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\metadata"
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "data_pipeline", "gemini_prompts")
RESULT_JSON_PATH = os.path.join(OUTPUT_DIR, "gemini_result.json")

# 每批发送的歌曲数量（API 模式建议小一些，避免超 token 限制）
API_BATCH_SIZE = 50


# ---- 复用 prepare_gemini_lrc_prompt.py 的工具函数 ----

def clean_lrc(text: str) -> str:
    """
    深度清洗 LRC 歌词：去时间轴、制作人员信息、元信息行、空行
    """
    cleaned = re.sub(r'^\{.*"c":\[.*\]\}$', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', cleaned)
    cleaned = re.sub(r'^\[(ar|ti|al|by|offset|hash|total|sign):.*\]$', '', cleaned,
                     flags=re.MULTILINE | re.IGNORECASE)
    cleaned = "\n".join([line.strip() for line in cleaned.split('\n') if line.strip()])
    return cleaned


def load_metadata(lrc_filename: str) -> Optional[Dict[str, str]]:
    """根据歌词文件名加载对应的元数据"""
    basename = lrc_filename.replace('.lrc', '')
    meta_path = os.path.join(METADATA_DIR, f"{basename}_meta.json")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        artists = "，".join([a[0] for a in meta.get("artist", [["Unknown"]])])
        duration_ms = meta.get("duration", 0)
        duration_str = f"{duration_ms // 60000}分{(duration_ms % 60000) // 1000}秒" if duration_ms else "未知"
        return {
            "title": meta.get("musicName", "Unknown"),
            "artist": artists,
            "album": meta.get("album", "Unknown"),
            "duration": duration_str,
        }
    except Exception:
        return None


def load_processed_files() -> set:
    """读取已处理的歌曲文件名集合"""
    if not os.path.exists(RESULT_JSON_PATH):
        return set()
    try:
        with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return {item.get("filename", "") for item in data if "filename" in item}
    except Exception as e:
        logger.warning(f"读取 gemini_result.json 时警告: {e}")
    return set()


def load_existing_results() -> List[Dict]:
    """读取已有的全部结果"""
    if not os.path.exists(RESULT_JSON_PATH):
        return []
    try:
        with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
            while raw and raw[-1] not in ']':
                raw = raw[:-1]
            data = json.loads(raw)
            return data if isinstance(data, list) else []
    except Exception:
        return []


# ---- Prompt 模板（与 prepare_gemini_lrc_prompt.py 完全一致）----

PROMPT_TEMPLATE = """你是 Neo4j 音乐知识图谱的构建专家。我需要你帮我分析以下 {count} 首歌曲的歌词，并严格提取出用于 Neo4j 图谱的多维度标签。

对于每一首歌，请你：
1. 深入理解歌词意境和整体氛围。
2. 提取 1~3 个核心情绪标签 (mood)。
   常见参考值："Happy", "Melancholy", "Healing", "Energetic", "Relaxing", "Romantic", "Nostalgic", "Angry", "Hopeful", "Dreamy", "Lonely", "Peaceful"
3. 提取 1~3 个探讨主题标签 (theme)。
   常见参考值："Love", "Heartbreak", "Growth", "Nature", "Youth", "Life", "Friendship", "Freedom", "Loss", "Self-discovery", "Night", "Journey"
4. 提取 1~2 个适合的收听场景标签 (scenario)。
   常见参考值："Study", "Driving", "Workout", "Sleep", "Commute", "Rainy Day", "Party", "Cooking", "Travel", "Late Night", "Morning", "Work"
5. 提取 1 个抽象氛围/美学标签 (vibe)。
   常见参考值："Lo-fi", "Cinematic", "Indie", "Acoustic", "Dreampop", "Urban", "Retro", "Ethereal", "Raw", "Warm", "Dark", "Bright"
6. 所有标签使用英文单词，保持专业和统一。不要发明已有近义词之外的新标签。

特殊情况处理：
- 如果歌词为纯音乐/器乐（无实质文字歌词），请将 moods 设为 ["Instrumental"]，themes 设为 []，scenario 和 vibe 仍然根据歌曲名和专辑信息尽力推断。
- 如果歌词极短或信息不足，请根据歌名、歌手风格和专辑名辅助推断。

请严格返回一个纯净的 JSON 数组。注意以下约束：
- 不要在 JSON 外包裹 markdown 代码块标记（如 ```json）
- 不要在 JSON 前后添加任何前言、解释或后语
- JSON 数组必须以 [ 开头，以 ] 结尾，] 后面不要有任何字符（包括句号、换行等）
- 确保外部脚本可以直接 json.loads 解析你的输出

数组里为每一首歌生成一个对象，格式如下：
[
    {{
        "filename": "光 - 陈粒.lrc",
        "moods": ["Melancholy", "Healing"],
        "themes": ["Love", "Life"],
        "scenarios": ["Late Night", "Commute"],
        "vibe": "Indie"
    }},
    ...
]

以下是这 {count} 首歌的歌词文本数据（已附带歌曲元信息供你参考）：
=============================================================

{lyrics_blocks}"""


def build_lyrics_blocks(lrc_files: List[Path]) -> str:
    """构建歌词内容文本块"""
    blocks = ""
    for file in lrc_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = clean_lrc(content)

            meta_info = load_metadata(file.name)
            meta_header = ""
            if meta_info:
                meta_header = (
                    f"  歌名: {meta_info['title']}\n"
                    f"  歌手: {meta_info['artist']}\n"
                    f"  专辑: {meta_info['album']}\n"
                    f"  时长: {meta_info['duration']}\n"
                )

            blocks += (
                f"【歌曲文件】: {file.name}\n"
                f"【基本信息】:\n{meta_header}"
                f"【歌词内容】:\n{content}\n\n"
                f"--------------------------\n"
            )
        except Exception as e:
            logger.warning(f"读取 {file.name} 失败: {e}")
    return blocks


def parse_llm_response(response_text: str) -> List[Dict]:
    """从 LLM 回复中提取 JSON 数组"""
    text = response_text.strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试去除 markdown 代码块
    if "```json" in text:
        json_start = text.find("```json") + 7
        json_end = text.find("```", json_start)
        text = text[json_start:json_end].strip()
    elif "```" in text:
        json_start = text.find("```") + 3
        json_end = text.find("```", json_start)
        text = text[json_start:json_end].strip()

    # 清洗尾部非法字符
    while text and text[-1] not in ']':
        text = text[:-1]

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        logger.error(f"原始回复前 500 字符: {response_text[:500]}")

    return []


def call_llm(prompt: str, provider: str, model: Optional[str], temperature: float) -> str:
    """调用 LLM API 获取回复"""
    from llms.multi_llm import MultiLLM

    llm = MultiLLM(provider=provider, model_name=model, temperature=temperature)

    logger.info(f"调用 LLM: provider={provider}, model={llm.model_name}")

    response = llm.invoke(
        system_prompt="你是一个专业的音乐标签提取专家。你只输出纯净 JSON，不输出任何额外文本。",
        user_prompt=prompt,
        max_tokens=8000,
    )

    return response


def save_results(all_results: List[Dict]) -> None:
    """保存结果到 gemini_result.json"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 已保存 {len(all_results)} 条结果到 {RESULT_JSON_PATH}")


def main():
    parser = argparse.ArgumentParser(description="歌词自动标签提取器（API 版）")
    parser.add_argument("--provider", default="siliconflow",
                        help="LLM 厂商: siliconflow, dashscope, google, deepseek, zhipu (默认: siliconflow)")
    parser.add_argument("--model", default=None,
                        help="指定模型名称（留空则使用厂商默认模型）")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="温度参数（标签提取建议低温，默认 0.3）")
    parser.add_argument("--batch-size", type=int, default=API_BATCH_SIZE,
                        help=f"每批处理歌曲数（默认 {API_BATCH_SIZE}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只展示待处理歌曲列表，不实际调用 API")
    args = parser.parse_args()

    # 1. 扫描待处理歌曲
    processed_files = load_processed_files()
    lrc_files = list(Path(LYRICS_DIR).glob("*.lrc"))
    pending = [f for f in lrc_files if f.name not in processed_files]

    print(f"\n📊 歌曲统计:")
    print(f"   总歌词文件: {len(lrc_files)}")
    print(f"   已处理:     {len(processed_files)}")
    print(f"   待处理:     {len(pending)}")

    if not pending:
        print(f"\n🎉 所有 {len(lrc_files)} 首歌词都已标注完成！")
        return

    if args.dry_run:
        print(f"\n📋 待处理歌曲列表:")
        for i, f in enumerate(pending, 1):
            print(f"   {i}. {f.name}")
        print(f"\n（使用 --dry-run=false 开始处理）")
        return

    # 2. 分批处理
    total_batches = math.ceil(len(pending) / args.batch_size)
    all_results = load_existing_results()
    new_count = 0

    print(f"\n🚀 开始处理，共 {total_batches} 批 (每批 {args.batch_size} 首)")
    print(f"   LLM: {args.provider}" + (f" / {args.model}" if args.model else " (默认模型)"))
    print()

    for batch_idx in range(total_batches):
        batch_files = pending[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
        batch_names = [f.name for f in batch_files]

        print(f"📦 批次 {batch_idx + 1}/{total_batches}: {len(batch_files)} 首歌曲")
        for name in batch_names:
            print(f"   - {name}")

        # 构建 prompt
        lyrics_blocks = build_lyrics_blocks(batch_files)
        prompt = PROMPT_TEMPLATE.format(count=len(batch_files), lyrics_blocks=lyrics_blocks)

        # 调用 LLM
        try:
            start_time = time.time()
            response = call_llm(prompt, args.provider, args.model, args.temperature)
            elapsed = time.time() - start_time

            # 解析结果
            batch_results = parse_llm_response(response)

            if batch_results:
                all_results.extend(batch_results)
                new_count += len(batch_results)
                save_results(all_results)
                print(f"   ✅ 成功提取 {len(batch_results)} 首标签 ({elapsed:.1f}s)\n")
            else:
                print(f"   ❌ 本批次解析失败，跳过\n")
                # 保存原始响应以便调试
                debug_path = os.path.join(OUTPUT_DIR, f"debug_batch_{batch_idx + 1}.txt")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"   💾 原始响应已保存至: {debug_path}")

        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
            print(f"   ❌ API 调用失败: {e}\n")

        # 批次间间隔（避免限流）
        if batch_idx < total_batches - 1:
            time.sleep(2)

    print(f"\n🎉 处理完成！本次新增 {new_count} 首歌曲标签")
    print(f"   结果文件: {RESULT_JSON_PATH}")
    print(f"   总标注数: {len(all_results)}")


if __name__ == "__main__":
    main()
