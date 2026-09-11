"""Opt-in, synthetic-only API smoke; not an end-to-end retrieval benchmark."""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
import json
import logging
import os
from pathlib import Path
import time

import httpx
from dotenv import dotenv_values


async def run_smoke():
    # Read only the project credential; never emit its value or forward to custom hosts.
    values = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    key = values.get('DASHSCOPE_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not key:
        return {'success': False, 'stage': 'credential_missing'}
    from agent.intent.adapters import PlannerPayload, plan_with_dashscope
    from llms.prompts import UNIFIED_PLANNER_HUMAN, UNIFIED_PLANNER_SYSTEM
    from schemas.tool_plan import tool_plan_alignment_issues
    logging.disable(logging.CRITICAL)
    model = 'qwen3.7-plus'
    base = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    report = {'model': model, 'date': str(date.today()), 'scope': 'synthetic_planner_and_prose_only',
              'not_tested': ['retrieval', 'memory_storage', 'playback', '35b_performance'], 'cases': []}
    previous = ''
    history = ''
    for query in ['外面下暴雨，想听安静但不压抑的音乐', '梦幻一点的歌曲有没有']:
        started = time.perf_counter()
        try:
            payload = PlannerPayload(user_input=query, user_preferences='', chat_history=history,
                                     previous_plan=previous, current_date=str(date.today()), retrieved_memories='')
            plan = await plan_with_dashscope(str(key), model, UNIFIED_PLANNER_SYSTEM, UNIFIED_PLANNER_HUMAN,
                                             payload, max_tokens=3000, timeout=60, base_url=base)
            issues = tool_plan_alignment_issues(plan)
            report['cases'].append({'query': query, 'schema_valid': True, 'alignment_issues': issues,
                                    'seconds': round(time.perf_counter() - started, 2),
                                    'plan': plan.model_dump(mode='json')})
            previous = plan.model_dump_json()
            history = json.dumps([{'role': 'user', 'content': query},
                                  {'role': 'assistant', 'content': '测试上一轮已返回歌单，可继续调整。'}], ensure_ascii=False)
        except Exception as exc:
            # API bodies and transport exceptions can contain sensitive information.
            report['success'] = False
            report['failure_type'] = type(exc).__name__
            return report
    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False, follow_redirects=False) as client:
            response = await client.post(base + '/chat/completions', headers={'Authorization': 'Bearer ' + str(key)},
                json={'model': model, 'enable_thinking': False, 'max_tokens': 160, 'temperature': 0,
                      'messages': [{'role': 'system', 'content': '你是音乐助手。只用提供的虚构测试曲目解释，不编造其他歌曲。'},
                                   {'role': 'user', 'content': '测试曲目：雨窗（轻柔环境音乐）。请用两句话介绍适合雨天放松的理由。'}]})
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            report['prose'] = {'nonempty': bool(str(content).strip()), 'text': content}
        report['success'] = report['prose']['nonempty'] and all(not c['alignment_issues'] for c in report['cases'])
    except Exception as exc:
        report.update(success=False, failure_type=type(exc).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Authorize up to five billed requests using synthetic text only')
    args = parser.parse_args()
    if not args.run:
        parser.error('No network call made. Use --run to execute the bounded API smoke.')
    try:
        result = asyncio.run(asyncio.wait_for(run_smoke(), timeout=310))
    except Exception as exc:
        result = {'success': False, 'failure_type': type(exc).__name__, 'stage': 'probe_aborted'}
    target = Path(__file__).resolve().parents[1] / 'tests/eval/results/release_api_smoke_20260911.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'success': result.get('success'), 'failure_type': result.get('failure_type'),
                      'scope': result.get('scope'), 'stage': result.get('stage'), 'cases': len(result.get('cases', [])),
                      'result_file': str(target)}, ensure_ascii=False))
    return 0 if result.get('success') else 1


if __name__ == '__main__':
    raise SystemExit(main())
