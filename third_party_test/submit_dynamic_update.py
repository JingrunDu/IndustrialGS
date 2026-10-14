"""Prepare or submit two isolated dynamic-update tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
MOUNT = Path('/data/oss_bucket_0')
START = 665000
ADDITIONAL = 6650 * 50


def build_plan(mount, run_name):
    tests = []
    for number, experiment in enumerate(['E01', 'E09'], 1):
        config = json.loads((REPO / 'experiments/next_16/configs' / (experiment + '.json')).read_text())
        source = mount / config['output_relative']
        checkpoint = source / 'checkpoints' / ('checkpoint_%d.pth' % START)
        if not checkpoint.is_file() or not Path(str(checkpoint) + '.sha256').is_file():
            raise FileNotFoundError('Missing verified checkpoint: %s' % checkpoint)
        manifest = REPO / 'experiments/next_16' / config['manifest']
        if config['num_images'] != 6650 or len(manifest.read_text().splitlines()) != 6650:
            raise ValueError('Expected the original 6,650-image dataset')
        output = mount / 'beifen/data/3dgs_bushu/dynamic_update' / run_name / ('test%d' % number)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Output is not empty: %s' % output)
        tests.append(dict(test=number, experiment=experiment, source=str(source),
                          checkpoint=str(checkpoint), output=str(output), num_images=6650,
                          start_iteration=START, additional_iterations=ADDITIONAL,
                          stop_iteration=START + ADDITIONAL))
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true', help='Default: validate and preview only')
    parser.add_argument('--run-name', default=time.strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a directory name')
    tests = build_plan(MOUNT, args.run_name)
    print(json.dumps(tests, ensure_ascii=False, indent=2), flush=True)
    if not args.submit:
        print('Preview only. Add --submit to launch two one-GPU tasks.')
        return
    for test in tests:
        env = dict(os.environ, EXPERIMENT_ID=test['experiment'], DYNAMIC_UPDATE='1',
                   RESUME_CHECKPOINT=test['checkpoint'], EXPERIMENT_OUTPUT=test['output'],
                   STOP_AFTER=str(test['stop_iteration']),
                   JOB_NAME='dynamic_update_test%d_%s' % (test['test'], args.run_name))
        subprocess.run(['bash', str(REPO / 'deployment/huanliufa_nebula/submit.sh')], env=env, check=True)


if __name__ == '__main__':
    main()
