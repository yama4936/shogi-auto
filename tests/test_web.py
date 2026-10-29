from http.cookiejar import CookieJar
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

from shogi_auto.web import ShogiServer


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ShogiServer(('127.0.0.1', 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.client = build_opener(HTTPCookieProcessor(CookieJar()))
        self.state = self.request('/api/game')

    def request(self, path, payload=None, status=200, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base + path, data=data, headers=headers or {'Content-Type': 'application/json'})
        try:
            response = self.client.open(request, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            self.assertEqual(response.status, status)
            return json.load(response)

    def post(self, path, **values):
        self.state = self.request(path, {'version': self.state['version'], **values})
        return self.state

    def test_full_roundtrip_stale_move_and_replay(self):
        self.post('/api/move', move='7g7f')
        self.post('/api/ai')
        self.assertEqual(len(self.state['history']), 2)
        self.assertEqual(self.state['turn'], 0)
        self.request('/api/move', {'version': 0, 'move': '2g2f'}, status=409)
        self.assertEqual(len(self.request('/api/game')['history']), 2)
        self.post('/api/resign')
        self.assertEqual(self.state['result']['reason'], '投了')
        self.post('/api/new', human=1)
        self.assertEqual(self.state['history'], [])
        self.post('/api/ai')
        self.assertEqual(self.state['turn'], 1)

    def test_isolation_bad_input_and_origin(self):
        self.post('/api/move', move='7g7f')
        other = build_opener(HTTPCookieProcessor(CookieJar()))
        with other.open(self.base + '/api/game') as response:
            self.assertEqual(json.load(response)['history'], [])
        self.request('/api/new', {'version': self.state['version'], 'human': 'bad'}, status=400)
        self.request('/api/new', {'version': self.state['version']}, status=403,
                     headers={'Content-Type': 'application/json', 'Origin': 'https://untrusted.invalid'})
        self.assertEqual(len(self.request('/api/game')['history']), 1)
        self.request('/api/move', {'version': self.state['version'], 'move': 'invalid'}, status=400)

    def test_assets_and_no_path_traversal(self):
        for path in ('/', '/app.js', '/style.css'):
            with self.client.open(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])
                self.assertGreater(len(response.read()), 200)
        self.request('/../pyproject.toml', status=404)


if __name__ == '__main__':
    unittest.main()
