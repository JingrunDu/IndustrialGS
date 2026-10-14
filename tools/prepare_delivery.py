"""Organize the two reviewed tests; retain removed material in local backups."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OSS = Path('/data/oss_bucket_0/beifen/data/3dgs_bushu')


def copy_checked(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_name(target.name + '.partial')
    digest = hashlib.sha256()
    with source.open('rb') as reader, partial.open('wb') as writer:
        for chunk in iter(lambda: reader.read(8 * 1024**2), b''):
            writer.write(chunk)
            digest.update(chunk)
    checksum = digest.hexdigest()
    if expected and checksum != expected:
        raise ValueError('Checksum mismatch: %s' % source)
    partial.rename(target)
    print('Saved: %s' % target.relative_to(ROOT), flush=True)
    return checksum


def main():
    archive = ROOT / 'backups' / ('historical_' + time.strftime('%Y%m%d_%H%M%S'))
    def preserve(path):
        if path.exists():
            target = archive / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    preserve(ROOT / 'result')
    plan = ROOT / 'experiments/next_16'
    for path in (plan / 'configs').glob('*.json'):
        if path.stem not in {'E01', 'E09'}:
            preserve(path)
    for path in (plan / 'manifests').glob('*.txt'):
        if path.name != 'images_6650.txt':
            preserve(path)
    for name in ['prepare_plan.py', 'review_665000', 'validation.json']:
        preserve(plan / name)
    materials = ROOT / 'third_party_test'
    for name in ['新增视角渲染', '新增视角渲染_第二批', '测试对比图', '渲染候选图']:
        preserve(materials / name)
    records = []
    for test, experiment in [('test1', 'E01_T01_seed0'), ('test2', 'E09_T01_seed1')]:
        for phase, source, iteration in [
            ('modeling', OSS / 'experiments_20260923' / experiment, 665000),
            ('dynamic_update', OSS / 'dynamic_update/20261005_173800' / test, 997500),
        ]:
            target = ROOT / 'result' / test / phase
            for name in ['cfg_args', 'cameras.json', 'experiment.json', 'command.json',
                         'code_hashes.json', 'loss_log.csv', 'runtime.json', 'train.log',
                         'training_state.json', 'wandb_run.json', 'dynamic_update_result.json']:
                if (source / name).is_file():
                    copy_checked(source / name, target / name)
            names = ['point_cloud/iteration_%d/point_cloud.ply' % iteration,
                     'checkpoints/checkpoint_%d.pth' % iteration]
            for name in names:
                checksum_file = Path(str(source / name) + '.sha256')
                expected = checksum_file.read_text().split()[0] if checksum_file.exists() else None
                checksum = copy_checked(source / name, target / name, expected)
                Path(str(target / name) + '.sha256').write_text(checksum + '\n')
            records.append(dict(test=test, phase=phase, iteration=iteration,
                                source=str(source), directory=str(target.relative_to(ROOT))))
    for source, name in [(materials / '渲染对比大图_14_11_29.jpg', 'modeling.jpg'),
                         (materials / '动态更新渲染/动态更新_GT渲染_6对_精选版.jpg', 'dynamic_update.jpg')]:
        copy_checked(source, ROOT / 'result/renderings' / name)
    env_source = OSS / 'env/fastgs_py310_v1.tar'
    manifest = json.loads(Path(str(env_source) + '.json').read_text())
    env_target = ROOT / 'packages' / env_source.name
    copy_checked(env_source, env_target, manifest['sha256'])
    copy_checked(Path(str(env_source) + '.json'), Path(str(env_target) + '.json'))
    (ROOT / 'result/index.json').write_text(json.dumps(records, indent=2))
    print('Historical material retained in: %s' % archive, flush=True)


if __name__ == '__main__':
    main()
