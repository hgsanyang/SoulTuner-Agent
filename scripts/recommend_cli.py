"""Run with python -m scripts.recommend_cli '梦幻一点的歌曲有没有'."""
import argparse
import asyncio
import json
import sys
import uuid
from contextlib import aclosing

from services.local_recommendation_client import recommend_events, RecommendationClientError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SoulTuner local API client (no separate recommendation pipeline)")
    parser.add_argument("query")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--profile-id", default="local_admin")
    parser.add_argument("--personal", action="store_true", help="Use personal mode; may persist personal feedback/memory")
    parser.add_argument("--web", action="store_true", help="Allow server-side online song discovery")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)

    async def run():
        events = recommend_events(
            args.query, base_url=args.base_url, session_id=args.session_id or str(uuid.uuid4()),
            profile_id=args.profile_id, personal=args.personal, web_search_enabled=args.web,
            timeout_seconds=args.timeout,
        )
        async with aclosing(events):
            async for event in events:
                print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        asyncio.run(run())
    except (RecommendationClientError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
