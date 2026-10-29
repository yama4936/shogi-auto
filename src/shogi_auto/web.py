"""Local, per-browser shogi server. No accounts or external services required."""
import argparse
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import secrets
import threading
import time
from urllib.parse import urlsplit

from .game import Game

MAX_GAMES = 128
SESSION_TTL = 6 * 60 * 60


class Session:
    def __init__(self):
        self.game = Game()
        self.lock = threading.Lock()
        self.accessed = time.monotonic()


class ShogiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.sessions = {}
        self.session_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type="application/json; charset=utf-8", cookie=None):
        body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, dict) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        if cookie:
            self.send_header("Set-Cookie", f"shogi_session={cookie}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def session(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            pass
        token = cookie["shogi_session"].value if "shogi_session" in cookie else None
        with self.server.session_lock:
            now = time.monotonic()
            for key in list(self.server.sessions):
                if now - self.server.sessions[key].accessed > SESSION_TTL:
                    del self.server.sessions[key]
            if token not in self.server.sessions:
                if len(self.server.sessions) >= MAX_GAMES:
                    raise ValueError("対局数の上限です。しばらく待ってから再接続してください。")
                token = secrets.token_urlsafe(32)
                self.server.sessions[token] = Session()
            session = self.server.sessions[token]
            session.accessed = now
            return token, session

    def same_origin(self):
        # Reject browser DNS rebinding and cross-site mutations for a local server.
        try:
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        except ValueError:
            return False
        allowed = {"localhost", "127.0.0.1", "::1", self.server.server_address[0]}
        if host not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).netloc != self.headers.get("Host") or
                       urlsplit(origin).scheme not in ("http", "https")):
            return False
        return self.headers.get("Sec-Fetch-Site") != "cross-site"

    def do_GET(self):
        if not self.same_origin():
            self.reply(403, {"error": "同じ接続元からアクセスしてください。"})
            return
        path = urlsplit(self.path).path
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in static:
            name, content_type = static[path]
            self.reply(200, files("shogi_auto").joinpath("static", name).read_bytes(), content_type)
        elif path == "/api/game":
            try:
                token, session = self.session()
                with session.lock:
                    self.reply(200, session.game.snapshot(), cookie=token)
            except ValueError as exc:
                self.reply(503, {"error": str(exc)})
        else:
            self.reply(404, {"error": "ページが見つかりません。"})

    def do_POST(self):
        if not self.same_origin():
            self.reply(403, {"error": "同じ接続元から操作してください。"})
            return
        path = urlsplit(self.path).path
        if path not in ("/api/move", "/api/ai", "/api/new", "/api/resign"):
            self.reply(404, {"error": "操作が見つかりません。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("JSON形式で操作してください。")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("操作の形式が正しくありません。")
            token, session = self.session()
            with session.lock:
                game = session.game
                if type(payload.get("version")) is not int or payload["version"] != game.version:
                    self.reply(409, {"error": "盤面が更新されています。再読み込みしてください。"}, cookie=token)
                    return
                if path == "/api/new":
                    human = payload.get("human", 0)
                    if type(human) is not int or human not in (0, 1):
                        raise ValueError("先手か後手を選んでください。")
                    session.game = Game(human)
                    session.game.version = game.version + 1
                elif path == "/api/move":
                    game.human_move(payload.get("move"))
                elif path == "/api/ai":
                    game.ai_move()
                else:
                    game.resign()
                self.reply(200, session.game.snapshot(), cookie=token)
        except (ValueError, UnicodeDecodeError, CookieError) as exc:
            self.reply(400, {"error": str(exc)})

    def log_message(self, format, *args):
        # Do not log cookies or game data.
        super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="人間とCPU AIの将棋対局（ローカル用）")
    parser.add_argument("--host", default="127.0.0.1", help="既定: 127.0.0.1。信頼できる接続内で使用")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with ShogiServer((args.host, args.port)) as server:
        print(f"将棋を開始: http://{args.host}:{server.server_port} （Ctrl+Cで終了）", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
