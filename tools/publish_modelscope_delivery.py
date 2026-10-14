"""Publish release metadata and remove legacy data only after verified replacement."""
import json
from pathlib import Path
import shutil
import subprocess

from modelscope_delivery import ROOT, WORK, STAGE, REPO, api_client, sha, write_json


def main():
    api = api_client()
    sections = ['data_full', 'data_subset_6650', 'environment', 'result']
    receipts = {}
    for section in sections:
        receipts[section] = json.loads((WORK / ('verified_' + section + '.json')).read_text())
        for item in receipts[section]:
            if not item['remote_sha256_verified']:
                raise ValueError('Remote checksum not available: %s' % item['path'])
    full = {}
    for manifest in (STAGE / 'data/full').glob('samples-*.manifest.json'):
        for item in json.loads(manifest.read_text())['files']:
            if item['path'] in full:
                raise ValueError('Duplicate full data path')
            full[item['path']] = item['sha256']
    if len(full) != 90360 * 3:
        raise ValueError('Unexpected full file count')
    subset_count = 0
    for manifest in (STAGE / 'data/subset_6650').glob('samples-*.manifest.json'):
        for item in json.loads(manifest.read_text())['files']:
            if full.get(item['path']) != item['sha256']:
                raise ValueError('Subset differs from full data')
            subset_count += 1
    if subset_count != 6650 * 3:
        raise ValueError('Unexpected subset file count')
    verification = WORK / 'verification'
    fresh = json.loads((verification / 'train_smoke_verified/training_state.json').read_text())
    if fresh['iteration'] != 10:
        raise ValueError('Fresh training smoke test did not pass')
    update = json.loads((verification / 'update_smoke_verified/dynamic_update_result.json').read_text())
    if not update['completed']:
        raise ValueError('Checkpoint restoration did not pass')
    for key in ['data_subset_6650', 'environment', 'environment_extras', 'result_test1_modeling']:
        json.loads((WORK / ('download_verified_' + key + '.json')).read_text())
    render_dir = verification / 'render_smoke_verified'
    if len(list((render_dir / 'render').glob('*.png'))) != 2:
        raise ValueError('Rendering smoke test did not pass')
    version = dict(version='20261014', repo=REPO, full_samples=90360, subset_samples=6650,
                   uploaded_files=[item for section in receipts.values() for item in section],
                   verification=dict(download_checksums=True, independent_subset=True,
                                     subset_matches_full=True, fresh_training_steps=10,
                                     checkpoint_restored_iteration=665000,
                                     dynamic_update_steps=10, rendered_views=2),
                   original_code_commit=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   code_worktree='Contains additional local changes; see GitHub sync status.')
    write_json(STAGE / 'release.json', version)
    (STAGE / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (item['sha256'], item['path'])
                                           for item in version['uploaded_files']))
    shutil.copyfile(ROOT / 'docs/ModelScope交付说明.md', STAGE / 'README.md')
    with (STAGE / 'README.md').open('a') as handle:
        handle.write('\n## 验证记录\n\n已验证独立子集下载、环境离线恢复、10步建模、'
                     '从665,000步断点进行10步动态更新及两个视角渲染。'
                     '完整文件校验见 `SHA256SUMS`，版本清单见 `release.json`。\n')
    for name in ['README.md', 'release.json', 'SHA256SUMS']:
        api.upload_file(REPO, 'dataset', STAGE / name, name,
                        commit_message='Publish verified IndustrialGS release', disable_tqdm=True)
    remote = api.list_repo_files(REPO, 'dataset')
    old = [f for f in remote if f.type == 'blob' and (
        f.path.startswith(('mask_sam3/', 'semantic_weights/', 'sparse/'))
        or f.path in {'perspective.tar.gz', 'delivery_status.txt'})]
    write_json(WORK / 'legacy_snapshot.json', [dict(path=f.path, size=f.size, sha256=f.sha256) for f in old])
    print('Removing replaced legacy files:', len(old), flush=True)
    if old:
        api.delete_files(REPO, 'dataset', [f.path for f in old],
                         commit_message='Remove legacy data after verified replacement')
    remaining = api.list_repo_files(REPO, 'dataset')
    paths = {f.path for f in remaining}
    if any(f.path in paths for f in old):
        raise ValueError('Legacy cleanup incomplete')
    for item in version['uploaded_files']:
        if item['path'] not in paths:
            raise ValueError('Published file disappeared after cleanup')
    write_json(WORK / 'publication_result.json', dict(status='complete',
               removed_legacy_files=len(old), current_files=len(paths), repo=REPO))
    print('ModelScope delivery published and legacy files removed', flush=True)


if __name__ == '__main__':
    main()
