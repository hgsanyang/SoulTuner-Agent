# -*- coding: utf-8 -*-
"""
prepare_sft_data.py
===================
将 planner_sft_data.jsonl (600条) 转换为 Unsloth 训练所需的 ChatML 格式，
并按 500/100 分层抽样划分训练集和验证集。

用法:
    python prepare_sft_data.py

输出:
    data/sft/train_chatml.jsonl  (500条)
    data/sft/eval_chatml.jsonl   (100条)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ===== 配置 =====
SEED = 42
EVAL_SIZE = 100  # 验证集大小
INPUT_FILE = Path(__file__).parent / "planner_sft_data.jsonl"
TRAIN_FILE = Path(__file__).parent / "train_chatml.jsonl"
EVAL_FILE = Path(__file__).parent / "eval_chatml.jsonl"

# ===== 精简版 System Prompt (~600 token) =====
# 保留: 角色定义、检索引擎说明、策略规则、意图枚举
# 删除: few-shot示例、chat_history/user_input占位符、冗长解释
CONDENSED_SYSTEM_PROMPT = """你是一个音乐推荐智能体的核心决策器。根据用户输入，一次性完成意图分析和检索规划，输出严格的JSON。

## 可用检索引擎
- **知识图谱 (Graph)**：存储 [歌手]-演唱->[歌曲], [歌曲]-主题/情绪/场景/语言/地区 等结构化关系
- **声学向量 (Vector)**：用 M2D-CLAP 将音频编码为向量，适合主观听感描述
- **联网搜索 (Web)**：适合时效性内容（最新专辑、演唱会、新闻）

注意：`vector_acoustic_query` 由下游模块自动生成，你只需设置 `use_vector: true/false`，将 `vector_acoustic_query` 留空。

## 策略规则
1. 具体歌手/乐队名 + 询问其歌曲 → `search`，启用 Graph
2. 主观声学/情绪/氛围描述 → 启用 Vector
3. 两者都有 → 同时启用 Graph + Vector
4. 涉及"最新/最近/新歌/演唱会/新闻" → 启用 Web
5. 纯情绪无实体 → `recommend_by_mood`
6. 明确流派无歌手 → `recommend_by_genre`
7. 活动场景 → `recommend_by_activity`，graph_scenario_filter 必填
8. 闲聊/非音乐问题 → `general_chat`，无需检索
9. 要下载/获取歌曲 → `acquire_music`，parameters 填 song_queries
10. 外语歌手别名翻译：graph_entities 同时含中英文名
11. 用户提到"我喜欢的/收藏的/点赞的" → `recommend_by_favorites`

## 意图类型
- `play_specific_song_online`：用户给出具体歌名要求播放
- `search`：搜索特定歌手/歌曲/音乐信息
- `acquire_music`：下载/获取/入库歌曲
- `recommend_by_mood`：按心情推荐
- `recommend_by_genre`：按流派推荐
- `recommend_by_artist`：按歌手推荐
- `recommend_by_favorites`：查询用户已收藏/点赞的歌曲
- `recommend_by_activity`：按场景活动推荐
- `general_chat`：闲聊

## 输出格式
严格输出JSON，包含字段：intent_type, parameters, context, retrieval_plan, reasoning。
retrieval_plan 包含：use_graph, graph_entities, graph_genre_filter, graph_scenario_filter, graph_mood_filter, graph_language_filter, graph_region_filter, use_vector, vector_acoustic_query, use_web_search, web_search_keywords。"""


def load_data(filepath):
    """加载 JSONL 数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"✅ 加载 {len(data)} 条数据 from {filepath}")
    return data


def to_chatml(sample):
    """将单条样本转换为 ChatML messages 格式"""
    user_input = sample["input"]
    assistant_output = json.dumps(sample["output"], ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": CONDENSED_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]
    }


def stratified_split(data, eval_size, seed):
    """
    分层抽样划分训练集和验证集
    确保每种意图类型在验证集中至少有 5 条
    """
    random.seed(seed)

    # 按意图类型分组
    groups = defaultdict(list)
    for item in data:
        intent = item["output"]["intent_type"]
        groups[intent].append(item)

    n_intents = len(groups)
    min_per_intent = max(5, eval_size // n_intents)  # 每种至少5条

    eval_set = []
    train_set = []

    for intent, items in groups.items():
        random.shuffle(items)
        # 计算该意图应分配的验证集条数（按比例，但至少 min_per_intent）
        proportional = max(min_per_intent, round(len(items) / len(data) * eval_size))
        # 不能超过该意图总数的 30%
        proportional = min(proportional, len(items) * 3 // 10)
        proportional = max(proportional, min(5, len(items)))

        eval_items = items[:proportional]
        train_items = items[proportional:]

        eval_set.extend(eval_items)
        train_set.extend(train_items)

    # 如果验证集不够 eval_size，从训练集中随机补充
    while len(eval_set) < eval_size and train_set:
        random.shuffle(train_set)
        eval_set.append(train_set.pop())

    # 如果验证集超过 eval_size，把多的放回训练集
    while len(eval_set) > eval_size:
        train_set.append(eval_set.pop())

    # 打乱顺序
    random.shuffle(train_set)
    random.shuffle(eval_set)

    return train_set, eval_set


def save_chatml(data, filepath):
    """将 ChatML 格式数据保存为 JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            chatml = to_chatml(item)
            f.write(json.dumps(chatml, ensure_ascii=False) + '\n')
    print(f"✅ 保存 {len(data)} 条 → {filepath}")


def print_distribution(data, label):
    """打印意图分布"""
    from collections import Counter
    intents = Counter(item["output"]["intent_type"] for item in data)
    print(f"\n📊 {label} ({len(data)} 条):")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"   {intent:35s} {count:4d} ({pct:5.1f}%)")


def main():
    print("=" * 60)
    print("🔄 Planner SFT 数据准备：JSONL → ChatML + 训练/验证拆分")
    print("=" * 60)

    # 1. 加载原始数据
    data = load_data(INPUT_FILE)

    # 2. 分层划分
    train_data, eval_data = stratified_split(data, EVAL_SIZE, SEED)

    # 3. 打印分布
    print_distribution(train_data, "训练集")
    print_distribution(eval_data, "验证集")

    # 4. 转换并保存 ChatML 格式
    save_chatml(train_data, TRAIN_FILE)
    save_chatml(eval_data, EVAL_FILE)

    # 5. 验证输出文件
    print("\n🔍 验证输出文件...")
    for fpath in [TRAIN_FILE, EVAL_FILE]:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    obj = json.loads(line)
                    assert "messages" in obj
                    assert len(obj["messages"]) == 3
                    assert obj["messages"][0]["role"] == "system"
                    assert obj["messages"][1]["role"] == "user"
                    assert obj["messages"][2]["role"] == "assistant"
                except Exception as e:
                    print(f"❌ 第 {i+1} 行格式错误: {e}")
                    return
        print(f"   ✅ {fpath.name}: {len(lines)} 条，格式全部合法")

    # 6. 打印 token 估算
    sample = to_chatml(data[0])
    total_chars = sum(len(m["content"]) for m in sample["messages"])
    est_tokens = total_chars // 2  # 粗略估算中文约 2 字符/token
    print(f"\n📐 单条样本估算 token: ~{est_tokens} (system + user + assistant)")
    print(f"   训练集总 token 估算: ~{est_tokens * len(train_data):,}")

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print(f"   训练集: {TRAIN_FILE} ({len(train_data)} 条)")
    print(f"   验证集: {EVAL_FILE} ({len(eval_data)} 条)")
    print("   下一步: 上传到 ModelScope 并运行 finetune_planner.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
