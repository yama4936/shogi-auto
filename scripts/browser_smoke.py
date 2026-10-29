"""Optional real Chromium smoke test; game fixtures are injected in-process only.

Install: .venv/bin/python -m pip install -e '.[browser]'
Run: .venv/bin/python scripts/browser_smoke.py --chromium /usr/bin/chromium
Screenshots go to ignored outputs/browser/. No network or external service used.
"""
import argparse
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright, expect

from shogi_auto.game import Game
from shogi_auto.web import ShogiServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', help='System Chromium path, or omit for Playwright-managed Chromium')
    args = parser.parse_args()
    output = Path('outputs/browser')
    output.mkdir(parents=True, exist_ok=True)
    with ShogiServer(('127.0.0.1', 0)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=args.chromium, headless=True, args=['--no-sandbox'])
                context = browser.new_context(viewport={'width': 1100, 'height': 1000})
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('dialog', lambda dialog: dialog.accept())

                def snapshot():
                    return context.request.get(base + '/api/game').json()

                def ready(count):
                    expect(page.locator('#move-count')).to_have_text(f'{count}手')
                    expect(page.locator('#new')).to_be_enabled()
                    expect(page.locator('#error')).to_be_empty()

                def move(text):
                    if '*' in text:
                        page.locator(f'#human-hand [data-drop="{text[0]}"]').click()
                    else:
                        page.locator(f'[data-square="{text[:2]}"]').click()
                    page.locator(f'[data-square="{text[2:4]}"]').click()

                def fixture(sfen=None, moves=(), human=0):
                    token = next(cookie['value'] for cookie in context.cookies() if cookie['name'] == 'shogi_session')
                    session = server.sessions[token]
                    with session.lock:
                        game = Game(human=human, sfen=sfen)
                        for text in moves:
                            game.push(text)
                        game.version = session.game.version + 1
                        session.game = game
                    page.reload()
                    ready(len(moves))

                page.goto(base)
                ready(0)
                expect(page.locator('.square')).to_have_count(81)
                page.locator('[data-square="7g"]').click()
                page.locator('[data-square="7e"]').click()
                expect(page.locator('#error')).to_contain_text('そのマスには指せません')
                expect(page.locator('#move-count')).to_have_text('0手')
                page.locator('[data-square="7f"]').click()
                ready(2)
                for count in (4, 6):
                    legal = snapshot()['legal']
                    move(next(text for text in legal if not text.endswith('+')))
                    ready(count)
                opening = snapshot()['history']
                page.screenshot(path=str(output / 'desktop.png'), full_page=True)
                page.reload(); ready(6)
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
                page.screenshot(path=str(output / 'mobile.png'), full_page=True)

                for promote in (True, False):
                    fixture(moves=('7g7f', '3c3d'))
                    move('8h2b')
                    expect(page.locator('#promotion')).to_be_visible()
                    page.locator('#cancel-promotion').click()
                    expect(page.locator('#move-count')).to_have_text('2手')
                    page.locator('[data-square="2b"]').click()
                    page.locator('#promote' if promote else '#unpromote').click()
                    ready(4)
                    assert snapshot()['history'][2]['usi'] == ('8h2b+' if promote else '8h2b')

                fixture(sfen='4k4/P8/9/9/9/9/9/9/4K4 b - 1')
                move('9b9a+')
                ready(2)
                expect(page.locator('#promotion')).not_to_be_visible()
                assert snapshot()['history'][0]['usi'] == '9b9a+'

                fixture(moves=('7g7f', '3c3d', '8h2b+', '3a2b'))
                move('B*4e'); ready(6)
                assert snapshot()['history'][4]['usi'] == 'B*4e'
                expect(page.locator('#human-hand [data-drop="B"]')).to_have_count(0)

                fixture(moves=('7g7f', '3c3d', '8h2b+', '4a5b', 'B*4b', '5a4a'))
                move('2b3a'); ready(7)
                expect(page.locator('#status')).to_contain_text('詰み')
                expect(page.locator('#resign')).to_be_disabled()
                expect(page.locator('.square:not([disabled])')).to_have_count(0)
                page.locator('#new').click(); ready(0)
                page.locator('#side').select_option('1')
                page.locator('#new').click(); ready(1)
                expect(page.locator('#human-label')).to_have_text('後手 · あなた')
                assert page.locator('.square').first.get_attribute('data-square') == '1i'
                page.locator('#resign').click()
                expect(page.locator('#status')).to_contain_text('投了')
                assert not errors, errors
                browser.close()
                print(json.dumps({'status': 'passed', 'opening_usi': [entry['usi'] for entry in opening],
                                  'checks': ['3 human moves and AI replies', 'reload', 'mobile layout',
                                             'illegal move', 'promotion and cancellation', 'forced promotion',
                                             'drop', 'checkmate', 'replay', 'human gote', 'resign'],
                                  'screenshots': str(output)}, ensure_ascii=False))
        finally:
            server.shutdown()
            thread.join()


if __name__ == '__main__':
    main()
