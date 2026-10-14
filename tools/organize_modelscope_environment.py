"""Move a verified environment into its own versioned directory and publish it."""
import json
import shutil

from modelscope_delivery import ENVIRONMENT, REPO, ROOT, STAGE, WORK, api_client, sha, upload, write_json


def main():
    base = STAGE / 'environment'
    target = base / ENVIRONMENT
    target.mkdir(exist_ok=True)
    for path in list(base.iterdir()):
        if path == target or (path.is_dir() and path.name != 'wheels'):
            continue
        destination = target / path.name
        if destination.exists():
            raise ValueError('Migration destination already exists: %s' % destination)
        shutil.move(str(path), str(destination))
    renamed = []
    for path in list(target.glob('fastgs_py310_v1.tar*')):
        destination = path.with_name(path.name.replace('fastgs_py310_v1', 'industrialgs_py310_v1', 1))
        if destination.exists():
            raise ValueError('Archive rename destination already exists')
        renamed.append(path.relative_to(STAGE).as_posix())
        path.rename(destination)
    parts_path = target / 'industrialgs_py310_v1.tar.parts.json'
    parts = json.loads(parts_path.read_text())
    parts['original'] = 'industrialgs_py310_v1.tar'
    for part in parts['parts']:
        part['file'] = part['file'].replace('fastgs_py310_v1', 'industrialgs_py310_v1', 1)
    write_json(parts_path, parts)
    archive_path = target / 'industrialgs_py310_v1.tar.json'
    archive = json.loads(archive_path.read_text())
    archive['archive'] = 'industrialgs_py310_v1.tar'
    write_json(archive_path, archive)
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size, sha256=sha(p))
             for p in sorted(target.rglob('*')) if p.is_file()
             and p.name not in {'README.md', 'manifest.json'}]
    write_json(target / 'manifest.json', dict(name=ENVIRONMENT, python='3.10',
               pytorch='2.3.0+cu121', platform='Linux x86_64, glibc >= 2.28', files=files))
    shutil.copyfile(ROOT / 'docs/ModelScope交付说明.md', target / 'README.md')
    section = 'environment/' + ENVIRONMENT
    upload(section)
    receipt = json.loads((WORK / ('verified_' + section.replace('/', '_') + '.json')).read_text())
    if not all(item['remote_sha256_verified'] for item in receipt):
        raise ValueError('New environment checksum verification incomplete')
    write_json(WORK / 'verified_environment.json', receipt)
    release = json.loads((STAGE / 'release.json').read_text())
    release['uploaded_files'] = [f for f in release['uploaded_files']
                                 if not f['path'].startswith('environment/')] + receipt
    release['environment'] = ENVIRONMENT
    write_json(STAGE / 'release.json', release)
    (STAGE / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (f['sha256'], f['path'])
                                           for f in release['uploaded_files']))
    shutil.copyfile(ROOT / 'docs/ModelScope交付说明.md', STAGE / 'README.md')
    api = api_client()
    for name in ['README.md', 'release.json', 'SHA256SUMS']:
        api.upload_file(REPO, 'dataset', STAGE / name, name, disable_tqdm=True,
                        commit_message='Organize environments by name and version')
    remote = {f.path: f for f in api.list_repo_files(REPO, 'dataset') if f.type == 'blob'}
    for name in ['README.md', 'release.json', 'SHA256SUMS']:
        if remote[name].sha256 != sha(STAGE / name):
            raise ValueError('Published metadata checksum differs')
    # Only remove known files from this environment's former layout.
    old = ['environment/' + f['path'] for f in files if 'environment/' + f['path'] in remote]
    old += [path for path in renamed if path in remote]
    write_json(WORK / 'environment_migration_old_files.json', old)
    if old:
        api.delete_files(REPO, 'dataset', old,
                         commit_message='Remove verified environment from former shared layout')
    final = {f.path: f for f in api.list_repo_files(REPO, 'dataset') if f.type == 'blob'}
    if any(path in final for path in old):
        raise ValueError('Old environment cleanup incomplete')
    for item in release['uploaded_files']:
        if item['path'] not in final or final[item['path']].sha256 != item['sha256']:
            raise ValueError('Final release checksum differs: %s' % item['path'])
    write_json(WORK / 'environment_migration_result.json', dict(status='complete',
               environment=ENVIRONMENT, files=len(receipt), removed_old_files=len(old),
               repository_files=len(final)))
    print('Environment migration verified and complete', flush=True)


if __name__ == '__main__':
    main()
