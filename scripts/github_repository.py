"""Create a private GitHub repository and push without persisting the access token."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--name', default='cowork-rsi')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.name):
        raise SystemExit('Invalid repository name.')
    token = args.token_file.read_text(encoding='utf-8-sig').strip()
    if not token or any(c.isspace() for c in token):
        raise SystemExit('Credential file must contain exactly one token; its contents were not logged.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}))

    def api(method: str, route: str, body: dict | None = None) -> tuple[int, dict]:
        request = urllib.request.Request('https://api.github.com' + route, method=method,
                    data=json.dumps(body).encode() if body else None,
                    headers={'Authorization': 'Bearer ' + token,
                             'Accept': 'application/vnd.github+json',
                             'X-GitHub-Api-Version': '2022-11-28',
                             'User-Agent': 'cowork-rsi-project-setup',
                             'Content-Type': 'application/json'})
        for attempt in range(4):
            try:
                with opener.open(request, timeout=30) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                if error.code in (502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                data = json.loads(error.read().decode('utf-8'))
                return error.code, {'message': str(data.get('message', '')).replace(token, '[redacted]')}
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise SystemExit('GitHub transport failed after four attempts; credential not logged.')
                time.sleep(2 ** attempt)
        raise RuntimeError('Unreachable')

    status, user = api('GET', '/user')
    if status != 200:
        raise SystemExit('GitHub identity check failed: ' + str(status) + ' ' + user.get('message', ''))
    login = user['login']
    owner_name = login + '/' + args.name
    status, repo = api('GET', '/repos/' + owner_name)
    if status == 404:
        status, repo = api('POST', '/user/repos', {'name': args.name, 'private': True,
                   'description': 'Reproducible professional Cowork agent baseline and feedback-driven procedural memory experiments.',
                   'auto_init': False})
        if status not in (200, 201):
            raise SystemExit('Repository creation failed: ' + str(status) + ' ' + repo.get('message', ''))
        created = True
    elif status == 200:
        created = False
    else:
        raise SystemExit('Repository lookup failed: ' + str(status) + ' ' + repo.get('message', ''))
    if repo.get('full_name', '').lower() != owner_name.lower() or not repo.get('private'):
        raise SystemExit('Existing destination does not match the requested private repository.')
    print(json.dumps({'repository': repo['full_name'], 'url': repo['html_url'], 'private': repo['private'],
                      'created': created}, ensure_ascii=False), flush=True)

    def git(*arguments: str, env: dict | None = None) -> None:
        subprocess.run(['git', '-C', str(ROOT), *arguments], check=True, env=env)

    if not (ROOT / '.git').exists():
        git('init', '-b', 'main')
        git('config', 'core.autocrlf', 'false')
        for key, default in [('user.name', login), ('user.email', f"{user['id']}+{login}@users.noreply.github.com")]:
            configured = subprocess.run(['git', '-C', str(ROOT), 'config', '--get', key],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if configured.returncode != 0:
                git('config', key, default)
    # Never persist token in the remote URL, config files, credential helpers or command arguments.
    existing = subprocess.run(['git', '-C', str(ROOT), 'remote', 'get-url', 'origin'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode != 0:
        git('remote', 'add', 'origin', repo['clone_url'])
    else:
        # An existing repository/remote must be reviewed by the caller before any push.
        current = subprocess.check_output(['git', '-C', str(ROOT), 'remote', 'get-url', 'origin'], text=True).strip()
        if current != repo['clone_url']:
            raise SystemExit('Existing origin differs from verified destination; refusing to replace it.')
    if args.push:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('GIT_TRACE', 'GIT_CONFIG_')) and key != 'GIT_CURL_VERBOSE'}
        credentials = base64.b64encode(('x-access-token:' + token).encode()).decode()
        env.update({'GIT_CONFIG_COUNT': '2',
                    'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
                    'GIT_CONFIG_VALUE_0': 'Authorization: Basic ' + credentials,
                    'GIT_CONFIG_KEY_1': 'http.proxy',
                    'GIT_CONFIG_VALUE_1': 'http://127.0.0.1:7897',
                    'GIT_TERMINAL_PROMPT': '0'})
        git('-c', 'credential.helper=', 'push', '-u', 'origin', 'main', env=env)
        print('PUSH_VERIFIED_BY_GIT', flush=True)


if __name__ == '__main__':
    main()
