"""Submit independent one-GPU experiments; default only prints the commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', choices=['E01', 'E09'], default=['E01', 'E09'])
    parser.add_argument('--stop-after', type=int, default=665000)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--resume-latest', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    launcher = root.parents[1] / 'deployment/huanliufa_nebula/submit.sh'
    for experiment in args.experiments:
        if experiment not in [f'E{i:02d}' for i in range(1, 17)]:
            raise ValueError(experiment)
        spec = json.loads((root / 'configs' / f'{experiment}.json').read_text())
        env = dict(os.environ, EXPERIMENT_ID=experiment, STOP_AFTER=str(args.stop_after),
                   QUEUE=os.environ.get('QUEUE', 'embodied_ai_h20_na130'),
                   JOB_NAME=f'huanliufa_{experiment}_{time.strftime("%Y%m%d_%H%M%S")}')
        if args.resume_latest:
            directory = Path('/data/oss_bucket_0') / spec['output_relative'] / 'checkpoints'
            candidates = [p for p in directory.glob('checkpoint_*.pth') if Path(str(p) + '.sha256').exists()]
            if not candidates:
                raise FileNotFoundError(f'No checkpoint: {directory}')
            env['RESUME_CHECKPOINT'] = str(max(candidates, key=lambda p: int(p.stem.split('_')[-1])))
        print(f'{experiment}: {spec["num_images"]} images, seed {spec["seed"]}, stop {args.stop_after}', flush=True)
        subprocess.run(['bash', str(launcher), *([] if args.submit else ['--dry-run'])], env=env, check=True)


if __name__ == '__main__':
    main()
