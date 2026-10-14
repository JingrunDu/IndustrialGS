import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deployment/huanliufa_nebula'))
from dynamic_update_node import summarize
from submit_dynamic_update import build_plan
from submit_with_auth import submit
from loss_tracking import track_loss_csv
import csv


class DynamicUpdateTests(unittest.TestCase):
    def test_local_loss_steps_preserve_cumulative_iteration(self):
        logged = []
        tracker = SimpleNamespace(log=logged.append)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'TRAIN_ITERATION_OFFSET': '665000'}):
            path = Path(tmp) / 'loss_log.csv'
            with path.open('w') as handle, track_loss_csv(tracker):
                writer = csv.writer(handle)
                writer.writerow(['iteration', 'total_loss'])
                writer.writerow([665100, 0.06])
            with path.open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ['iteration', 'total_loss', 'cumulative_iteration'])
            self.assertEqual(rows[1], ['100', '0.06', '665100'])
            self.assertEqual(logged, [{'train/iteration': 100, 'train/loss': 0.06}])

    def test_submission_packages_update_entry_and_training_code(self):
        def inspect(command, cwd, env):
            package = Path(cwd)
            self.assertTrue((package / 'dynamic_update_node.py').exists())
            self.assertTrue((package / 'dynamic_update_visualize.py').exists())
            self.assertTrue((package / 'render_comparison.py').exists())
            self.assertTrue((package / 'training_code/train.py').exists())
            self.assertTrue((package / 'training_code/experiments/next_16/manifests/images_6650.txt').exists())
            self.assertFalse((package / 'training_code/third_party_test').exists())
            return SimpleNamespace(returncode=0)
        with patch.dict('os.environ', {'EXPERIMENT_ID': 'E01', 'WANDB_MODE': 'disabled'}), patch('submit_with_auth.subprocess.run', inspect):
            self.assertEqual(submit(['nebulactl', 'run', 'mdl', '--entry=dynamic_update_node.py', '--file.cluster_file=cluster.json']), 0)

    def test_exact_checkpoint_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp)
            for name in ['E01_T01_seed0', 'E09_T01_seed1']:
                folder = mount / 'beifen/data/3dgs_bushu/experiments_20260923' / name / 'checkpoints'
                folder.mkdir(parents=True)
                (folder / 'checkpoint_665000.pth').touch()
                (folder / 'checkpoint_665000.pth.sha256').touch()
                (folder / 'checkpoint_997500.pth').touch()
            tests = build_plan(mount, 'test_run')
            self.assertEqual(len(tests), 2)
            self.assertEqual(tests[0]['additional_iterations'], 332500)
            self.assertEqual(tests[1]['stop_iteration'], 997500)
            self.assertTrue(tests[0]['checkpoint'].endswith('checkpoint_665000.pth'))
            self.assertNotEqual(tests[0]['output'], tests[0]['source'])
            output = Path(tests[0]['output'])
            output.mkdir(parents=True)
            (output / 'existing.txt').touch()
            with self.assertRaises(FileExistsError):
                build_plan(mount, 'test_run')

    def test_stage_time_and_incomplete_run_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'train.log').write_text('Training time: 4200.5\n')
            (output / 'training_state.json').write_text(json.dumps(dict(iteration=997500, gaussians=3300000)))
            result = summarize(output, 0)
            self.assertEqual(result['training_time_seconds'], 4200.5)
            self.assertEqual(result['additional_iterations'], 332500)
            self.assertTrue(result['meets_three_hours'])
            self.assertFalse(summarize(output, 1)['meets_three_hours'])
            (output / 'training_state.json').write_text(json.dumps(dict(iteration=700000)))
            self.assertFalse(summarize(output, 0)['meets_three_hours'])
            (output / 'train.log').write_text('No final timer\n')
            self.assertFalse(summarize(output, 0)['completed'])


if __name__ == '__main__':
    unittest.main()
