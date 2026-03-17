import os
import re
import json
from pathlib import Path
import math

# ---- 配置 ----
# 读取歌词的目录
LYRICS_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\lyrics"
# 读取元数据的目录（用来给 Gemini 提供歌曲背景信息）
METADATA_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio\metadata"
# 输出 prompt txt 的目录
OUTPUT_DIR = r"C:\Users\sanyang\sanyangworkspace\music_recommendation\Muisc-Research\data_pipeline\gemini_prompts"
# 每次打包几首歌给 Gemini（取决于上下文长度，Gemini 1.5 Pro 支持 100 万 token 上下文，可轻松处理 80~150 首）
BATCH_SIZE = 100

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
6. 判断歌词的主要语言 (language)，必须从以下固定值中选一个：
   "Chinese", "English", "Japanese", "Korean", "Cantonese", "Instrumental", "Mixed"
   - 判断依据是歌词的主体语言，例如普通话/国语统一标注为 "Chinese"
   - 粤语歌词（如广东话）请标注为 "Cantonese"
   - 如果是完全没有歌词的纯音乐，请标注为 "Instrumental"
   - 如果一首歌中英混杂比较均匀，请标注为 "Mixed"
7. 判断歌曲所属的音乐市场/地区 (region)，必须从以下固定值中选一个：
   "Mainland China", "Taiwan", "Hong Kong", "Japan", "Korea", "Western", "Other"
   - 判断依据是歌手来源和歌曲风格，如陈粒/许嵩/梁博等属于 "Mainland China"
   - 邓紫棋(G.E.M.)、张学友等港乐属于 "Hong Kong"
   - 周杰伦、陈奕迅(台)、五月天等台湾乐手属于 "Taiwan"
   - 欧美歌手（如 Taylor Swift, Current Joys, Benson Boone 等）属于 "Western"
8. 所有标签使用英文单词，保持专业和统一。不要发明已有近义词之外的新标签。

特殊情况处理：
- 如果歌词为纯音乐/器乐（无实质文字歌词），请根据歌曲名、歌手风格和专辑名尽力推断其 moods（如 "Peaceful"、"Relaxing"、"Dreamy"、"Melancholy" 等真实情绪）、themes（如有则填，否则 []）、scenarios 和 vibe。language 固定设为 "Instrumental"。【重要】切勿将 "Instrumental" 放入 moods 列表——它是语言/类型属性而非情绪描述词。
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
        "vibe": "Indie",
        "language": "Chinese",
        "region": "Mainland China"
    }},
    {{
        "filename": "A Different Age - Current Joys.lrc",
        "moods": ["Nostalgic", "Melancholy"],
        "themes": ["Youth", "Life"],
        "scenarios": ["Late Night", "Driving"],
        "vibe": "Indie",
        "language": "English",
        "region": "Western"
    }},
    ...
]

以下是这 {count} 首歌的歌词文本数据（已附带歌曲元信息供你参考）：
=============================================================

{lyrics_blocks}
"""

def clean_lrc(text):
    """
    深度清洗 LRC 歌词：
    1. 去掉时间轴标签 [00:15.33]
    2. 去掉网易云嵌入的 JSON 制作人员信息块 (如 {"t":204407,"c":[...]})
    3. 去掉常见的元信息行（如 [ar:xxx], [ti:xxx], [by:xxx]）
    4. 删掉所有空行，紧凑化
    """
    # 移除 JSON 格式的制作人员信息行（网易云 LRC 特有）
    cleaned = re.sub(r'^\{.*"c":\[.*\]\}$', '', text, flags=re.MULTILINE)
    # 移除时间轴标签 [xx:xx.xx]
    cleaned = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', cleaned)
    # 移除 LRC 元信息标签 [ar:xxx] [ti:xxx] [al:xxx] [by:xxx] [offset:xxx]
    cleaned = re.sub(r'^\[(ar|ti|al|by|offset|hash|total|sign):.*\]$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    # 删掉所有空行，紧凑化
    cleaned = "\n".join([line.strip() for line in cleaned.split('\n') if line.strip()])
    return cleaned

def load_metadata(lrc_filename):
    """
    根据歌词文件名找到对应的 _meta.json，
    提取歌名、歌手、专辑、时长等辅助信息供 Gemini 参考。
    """
    basename = lrc_filename.replace('.lrc', '')
    meta_path = os.path.join(METADATA_DIR, f"{basename}_meta.json")
    
    if not os.path.exists(meta_path):
        return None
        
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 提取关键字段
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

def generate_prompts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 【防重复机制】读取已有的 gemini_result.json，收集已经处理过的歌词文件名
    processed_files = set()
    result_json_path = os.path.join(OUTPUT_DIR, "gemini_result.json")
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "filename" in item:
                            processed_files.add(item["filename"])
            print(f"✅ 发现已解析的记录，将跳过这 {len(processed_files)} 首歌曲。")
        except Exception as e:
            print(f"读取 gemini_result.json 时警告 (不影响执行): {e}")

    lrc_files = list(Path(LYRICS_DIR).glob("*.lrc"))
    
    # 过滤掉已经处理过的歌词
    pending_lrc_files = [f for f in lrc_files if f.name not in processed_files]
    
    if not pending_lrc_files:
        print(f"🎉 太棒了，所有的 {len(lrc_files)} 首歌词都已经有提取结果了，无需再次生成！")
        return
        
    print(f"找到 {len(pending_lrc_files)} 首需要提取的歌曲 (总共 {len(lrc_files)} 首)，准备打包...")
    
    # 分批次打包
    total_batches = math.ceil(len(pending_lrc_files) / BATCH_SIZE)
    
    for i in range(total_batches):
        batch_files = pending_lrc_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        
        lyrics_blocks = ""
        for file in batch_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                content = clean_lrc(content)
                
                # 尝试加载元数据，为 Gemini 提供更多上下文
                meta_info = load_metadata(file.name)
                meta_header = ""
                if meta_info:
                    meta_header = (
                        f"  歌名: {meta_info['title']}\n"
                        f"  歌手: {meta_info['artist']}\n"
                        f"  专辑: {meta_info['album']}\n"
                        f"  时长: {meta_info['duration']}\n"
                    )
                
                lyrics_blocks += (
                    f"【歌曲文件】: {file.name}\n"
                    f"【基本信息】:\n{meta_header}"
                    f"【歌词内容】:\n{content}\n\n"
                    f"--------------------------\n"
                )
            except Exception as e:
                print(f"读取 {file.name} 失败: {e}")
                
        final_prompt = PROMPT_TEMPLATE.format(count=len(batch_files), lyrics_blocks=lyrics_blocks)
        
        output_file = os.path.join(OUTPUT_DIR, f"gemini_prompt_batch_{i+1}.txt")
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write(final_prompt)
            
        print(f"✅ 批次 {i+1}/{total_batches} 已生成: {output_file} (包含 {len(batch_files)} 首)")
        
    print("\n🎉 全部打包完成！")
    print("👉 下一步：请打开 Gemini 网页版 (https://gemini.google.com/)")
    print(f"👉 将 {OUTPUT_DIR} 里的 txt 内容复制发给它，并将它回复的 JSON 保存下来！")

if __name__ == "__main__":
    generate_prompts()
