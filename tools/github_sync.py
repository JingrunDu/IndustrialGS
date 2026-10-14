"""Authenticate Git without storing a token in its remote URL or configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TOKEN = Path.home() / '.github/huanliufa_upload.token'
REPO = 'JingrunDu/IndustrialGS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['auth', 'create', 'fetch', 'push', 'push-python', 'lfs-push', 'verify', 'credential'])
    parser.add_argument('operation', nargs='?')
    parser.add_argument('--directory', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == 'credential':
        fields = dict(line.rstrip('\n').split('=', 1) for line in sys.stdin if '=' in line)
        if args.operation == 'get' and fields.get('host') == 'github.com' and fields.get('protocol') == 'https':
            print('username=JingrunDu\npassword=' + TOKEN.read_text().strip() + '\n')
        return
    token = TOKEN.read_text().strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError('Token file is empty or contains whitespace')
    if args.action == 'push-python':
        from dulwich import porcelain
        result = porcelain.push(str(args.directory), 'https://github.com/' + REPO + '.git',
                                refspecs=['refs/heads/main:refs/heads/main'],
                                username='JingrunDu', password=token, set_upstream=True)
        if any(result.ref_status.values()):
            raise RuntimeError('GitHub rejected a pushed reference')
        return
    if args.action in {'auth', 'create'}:
        url = 'https://api.github.com/repos/' + REPO
        data = None
        if args.action == 'create':
            url = 'https://api.github.com/user/repos'
            data = json.dumps(dict(name='IndustrialGS', private=True, auto_init=False,
                              description='IndustrialGS training, dynamic update and rendering')).encode()
        request = urllib.request.Request(url, data=data,
                    headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                             'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                repo = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('GitHub authentication check returned HTTP %s' % error.code) from None
        print('Authenticated repository:', repo['full_name'])
        print('Private:', repo['private'])
        print('Repository push access:', repo.get('permissions', {}).get('push'))
        return
    helper = '!' + ' '.join(shlex.quote(arg) for arg in
                           [sys.executable, str(Path(__file__).resolve()), 'credential'])
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    for key in ['GIT_ASKPASS', 'SSH_ASKPASS']:
        env.pop(key, None)
    git = ['git', '-c', 'credential.helper=', '-c', 'credential.helper=' + helper,
           '-c', 'core.askPass=', '-c', 'http.version=HTTP/1.1', '-c', 'lfs.locksverify=false']
    command = {'fetch': ['fetch', 'origin', 'main'],
               'push': ['push', '--progress', '-u', 'origin', 'main'],
               'lfs-push': ['lfs', 'push', '--all', 'origin', 'main'],
               'verify': ['ls-remote', 'https://github.com/' + REPO + '.git', 'refs/heads/main']}[args.action]
    subprocess.run([*git, *command], cwd=args.directory, env=env, check=True, timeout=600)


if __name__ == '__main__':
    main()
