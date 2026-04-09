"""
意图分类评测脚本

用途：批量运行 Planner（LLM 意图分类），与标注数据对比，计算准确率。
需要：LLM API Key（会调用真实 LLM）

运行方式：
    # 使用默认 API 模型
    python -m tests.eval.evaluate_intent

    # 指定提供商
    python -m tests.eval.evaluate_intent --provider siliconflow

输出：
    - 总准确率
    - 每类意图的分类报告
    - 错误案例详情
    - Token 消耗统计
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


async def evaluate_intent(
    provider: str = "siliconflow",
    test_file: Optional[str] = None,
    verbose: bool = True,
):
    """
    运行意图分类评测

    Args:
        provider: LLM 提供商 (siliconflow / volcengine / gemini 等)
        test_file: 测试数据文件路径（默认使用 intent_test_queries.json）
        verbose: 是否打印每条详情
    """
    from llms.multi_llm import get_intent_chat_model
    from schemas.query_plan import MusicQueryPlan
    from langchain_core.prompts import ChatPromptTemplate
    from retrieval.gssc_context_builder import estimate_tokens

    # 加载测试数据
    if test_file is None:
        test_file = Path(__file__).parent / "intent_test_queries.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 初始化 LLM
    print(f"\n{'='*60}")
    print("SoulTuner-Agent 意图分类评测")
    print(f"{'='*60}")
    print(f"Provider: {provider}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'='*60}\n")

    from config.settings import settings
    settings.intent_llm_provider = provider

    llm = get_intent_chat_model()

    # 使用 API 大模型的 Planner Prompt
    from llms.prompts import UNIFIED_PLANNER_SYSTEM, UNIFIED_PLANNER_HUMAN
    structured_llm = llm.with_structured_output(MusicQueryPlan, include_raw=True)
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", UNIFIED_PLANNER_SYSTEM),
            ("human", UNIFIED_PLANNER_HUMAN),
        ])
        | structured_llm
    )

    # 运行评测
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected_intent"]
        case_id = case.get("id", f"case_{i}")

        try:
            t0 = time.time()
            raw_result = await chain.ainvoke({
                "user_input": query,
                "chat_history": "",
            })
            latency = time.time() - t0
            plan = raw_result["parsed"]
            predicted = plan.intent_type

            # 从 API 返回值提取真实 Token 消耗
            prompt_tok = 0
            completion_tok = 0
            raw_msg = raw_result.get("raw")
            if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                prompt_tok = raw_msg.usage_metadata.get("input_tokens", 0)
                completion_tok = raw_msg.usage_metadata.get("output_tokens", 0)
            total_prompt_tokens += prompt_tok
            total_completion_tokens += completion_tok
            total_latency += latency

            correct = predicted == expected
            results.append({
                "id": case_id,
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "latency": round(latency, 2),
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "reasoning": plan.reasoning[:80] if plan.reasoning else "",
            })

            status = "✅" if correct else "❌"
            if verbose:
                print(f"  [{case_id}] {status} {query[:30]:30s} → {predicted:25s} (expected: {expected})")
                if not correct:
                    print(f"           ⚠️  Reasoning: {plan.reasoning[:100]}")

        except Exception as e:
            results.append({
                "id": case_id,
                "query": query,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "latency": 0,
                "reasoning": str(e)[:100],
            })
            if verbose:
                print(f"  [{case_id}] 💥 ERROR: {query[:30]:30s} → {str(e)[:50]}")

    # ---- 统计汇总 ----
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\n{'='*60}")
    print("📊 评测结果汇总")
    print(f"{'='*60}")
    print(f"总准确率: {correct_count}/{total_count} = {accuracy:.1%}")
    print(f"平均延迟: {total_latency / max(total_count, 1):.2f}s")
    avg_prompt = total_prompt_tokens / max(total_count, 1)
    avg_completion = total_completion_tokens / max(total_count, 1)
    print(f"API Token 消耗: prompt={total_prompt_tokens:,} + completion={total_completion_tokens:,} = {total_prompt_tokens + total_completion_tokens:,} total")
    print(f"平均每次: prompt≈{avg_prompt:.0f} + completion≈{avg_completion:.0f} = {avg_prompt + avg_completion:.0f} tokens/query")
    print()

    # 按意图类型分类统计
    intent_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        intent_stats[r["expected"]]["total"] += 1
        if r["correct"]:
            intent_stats[r["expected"]]["correct"] += 1

    print("按意图类型分类:")
    print(f"  {'Intent Type':30s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>10s}")
    print(f"  {'-'*56}")
    for intent, stats in sorted(intent_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {intent:30s} {stats['correct']:>8d} {stats['total']:>6d} {acc:>10.1%}")

    # 打印错误案例
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n❌ 错误案例 ({len(errors)} 条):")
        for e in errors:
            print(f"  [{e['id']}] Query: {e['query']}")
            print(f"           Expected: {e['expected']} → Got: {e['predicted']}")
            print(f"           Reasoning: {e['reasoning']}")
            print()

    # 保存结果
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"intent_eval_{provider}_{timestamp}.json"

    report = {
        "provider": provider,
        "timestamp": timestamp,
        "total_cases": total_count,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "avg_latency_s": round(total_latency / max(total_count, 1), 2),
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_prompt_per_query": round(total_prompt_tokens / max(total_count, 1)),
            "avg_completion_per_query": round(total_completion_tokens / max(total_count, 1)),
        },
        "intent_stats": dict(intent_stats),
        "details": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细结果已保存至: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="SoulTuner 意图分类评测")
    parser.add_argument("--provider", default="siliconflow", help="LLM 提供商")
    parser.add_argument("--test-file", default=None, help="测试数据文件路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    asyncio.run(evaluate_intent(
        provider=args.provider,
        test_file=args.test_file,
        verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()
