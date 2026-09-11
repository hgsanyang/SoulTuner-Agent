"""Optional real Gradio component/callback smoke; no server or model launched."""
import asyncio
from importlib.util import find_spec
import os
import unittest
from unittest.mock import patch

from integrations.gradio_app import build_app
from services.presentation_session import PresentationSession

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

@unittest.skipUnless(find_spec('gradio'), 'Install optional requirements-local-ui.txt')
class LocalAppSmoke(unittest.TestCase):
    def test_real_component_build_and_stream_callback(self):
        app = build_app()
        submit = next(block.fn for block in app.fns.values() if block.fn and block.fn.__name__ == 'submit')
        self.assertTrue(any(row['type'] == 'chatbot' for row in app.get_config_file()['components']))

        async def events(query, **kwargs):
            yield {'type': 'recommendations_start'}
            yield {'type': 'song', 'song': {'title': 'A', 'audio_url': 'https://example.org/a.mp3'}}
            yield {'type': 'song', 'song': {'title': 'B'}}
            yield {'type': 'response', 'text': 'first'}
            yield {'type': 'response', 'text': 'finished'}
            yield {'type': 'complete', 'success': True}

        async def run():
            with patch('services.presentation_session.recommend_events', events):
                result = [row async for row in submit('rain', PresentationSession())]
            self.assertTrue(all(len(row) == 8 for row in result))
            # Player is populated once; subsequent songs/prose do not interrupt it.
            players = [row[6] for row in result if isinstance(row[6], str)]
            self.assertEqual(len(players), 1)
            self.assertIn('<audio controls', players[0])
            self.assertTrue(result[-1][4]['interactive'])
            self.assertTrue(result[-1][5]['interactive'])
        try:
            asyncio.run(run())
        finally:
            app.close()


if __name__ == '__main__':
    unittest.main()
