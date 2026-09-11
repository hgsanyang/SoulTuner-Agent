import asyncio
from pathlib import Path

import pytest
import yaml

from scripts import release_api_smoke


def test_live_probe_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr('sys.argv', ['release_api_smoke'])
    with pytest.raises(SystemExit) as error:
        release_api_smoke.main()
    assert error.value.code == 2


def test_live_probe_without_credential_does_not_send_requests(monkeypatch):
    monkeypatch.setattr(release_api_smoke, 'dotenv_values', lambda path: {})
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    assert asyncio.run(release_api_smoke.run_smoke()) == {'success': False, 'stage': 'credential_missing'}


def test_optional_ci_lanes_remain_separate_from_model_services():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[2] / '.github/workflows/ci.yml').read_text())
    jobs = workflow['jobs']
    assert {'music-mcp', 'local-gradio', 'neo4j-integration'} <= jobs.keys()
    for name in ['music-mcp', 'local-gradio', 'neo4j-integration']:
        assert 'secrets.' not in str(jobs[name])
        assert 'release_api_smoke' not in str(jobs[name])
    neo = jobs['neo4j-integration']['services']['neo4j']
    assert neo['ports'] == ['27687:7687']
    assert 'volumes' not in neo
