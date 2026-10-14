"""Resume a fixed test stage, record stage-only timing, then render comparisons."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from environment_archive import DEFAULT_ARCHIVE, restore_environment
from experiment_node import main as train_main

START = 665000
STOP = 997500


def summarize(output, exit_code, start_iteration=START, stop_iteration=STOP):
    output = Path(output)
    state = json.loads((output / 'training_state.json').read_text()) if (output / 'training_state.json').exists() else {}
    log = (output / 'train.log').read_text(errors='replace') if (output / 'train.log').exists() else ''
    matches = re.findall(r'^Training time:\s*([0-9.eE+-]+)', log, re.MULTILINE)
    seconds = float(matches[-1]) if matches else None
    iteration = state.get('iteration', start_iteration)
    complete = exit_code == 0 and iteration == stop_iteration and seconds is not None
    return dict(exit_code=exit_code, start_iteration=start_iteration, stop_iteration=iteration,
                additional_iterations=iteration-start_iteration, expected_additional_iterations=stop_iteration-start_iteration,
                training_time_seconds=seconds, timing='Training time: CUDA accumulated for this process only',
                gaussians=state.get('gaussians'), completed=complete,
                meets_three_hours=complete and seconds <= 3*3600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=['E01', 'E09'])
    parser.add_argument('--stop-after', type=int, required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stop_after != STOP or Path(args.resume).name != 'checkpoint_665000.pth':
        parser.error('Expected checkpoint 665000 and stop-after 997500')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dynamic_update_plan.json').write_text(json.dumps(dict(
        source_checkpoint=args.resume, num_images=6650, same_images_as_initial_training=True,
        start_iteration=START, additional_iterations=STOP-START, stop_iteration=STOP,
        timing='Training time resets to zero in each train.py process'), indent=2))
    # experiment_node rejects occupied outputs on fresh starts, but this is a resume.
    rc = 1
    os.environ['TRAIN_ITERATION_OFFSET'] = str(START)
    try:
        rc = train_main()
    finally:
        report = summarize(output, rc)
        (output / 'dynamic_update_result.json').write_text(json.dumps(report, indent=2))
        print('[动态更新结果]', json.dumps(report), flush=True)
    if rc or not report['completed']:
        return rc or 1
    here = Path(__file__).resolve().parent
    code = here / 'training_code' if (here / 'training_code').exists() else here.parents[1]
    mount = Path(os.environ.get('OSS_MOUNT', '/data/oss_bucket_0'))
    python = os.environ.get('GS_PYTHON') or restore_environment(mount / os.environ.get('ENV_ARCHIVE_REL', DEFAULT_ARCHIVE), '/tmp/fmc/fastgs')
    source = os.environ.get('DATA_PATH', str(mount / 'beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective'))
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    # These stages run after Training time and runtime.json have been recorded.
    for label, model in [('before', Path(args.resume).parent.parent), ('after', output)]:
        subprocess.run([python, '-u', str(here / 'render_comparison.py'), '--model', str(model), '--data', source,
                        '--project', str(code), '--output', str(output / 'visualization' / label),
                        '--per-camera', '6'], env=env, check=True)
    subprocess.run([python, str(here / 'dynamic_update_visualize.py'),
                    str(output / 'visualization')], env=env, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
