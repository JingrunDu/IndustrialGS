import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lazy_cameras import CameraCache, LazyCamera
from submit_with_auth import submit


class Tensor:
    def numel(self):
        return 8

    def element_size(self):
        return 4


class RuntimeTests(unittest.TestCase):
    def test_relative_initial_cloud_uses_selected_dataset(self):
        from experiment_node import command_for
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'downloaded_dataset'
            cloud = data / 'sparse/0/points3D.ply'
            cloud.parent.mkdir(parents=True)
            cloud.write_bytes(b'initial-cloud')
            manifest = root / 'images.txt'
            manifest.write_text('camera/image.jpg\n')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(manifest='images.txt', num_images=1, seed=0,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                parameters=dict(init_ply='sparse/0/points3D.ply',
                    init_ply_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest()))))
            spec, command = command_for(config, root, data, root / 'output', root, 10)
            self.assertEqual(command[command.index('--source_path') + 1], str(data))
            self.assertEqual(command[command.index('--init_ply') + 1], str(cloud))
            cloud.write_bytes(b'different-cloud')
            with self.assertRaisesRegex(ValueError, 'Initial point cloud checksum mismatch'):
                command_for(config, root, data, root / 'output', root, 10)

    def test_equal_semantic_weights_still_exclude_people(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import numpy as np
        from PIL import Image
        from camera_loading import load_mask
        import scene
        from utils.camera_utils import load_semantic_weight
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('mask_sam3', 'semantic_weights'):
                (root / name / 'camera').mkdir(parents=True)
            Image.fromarray(np.array([[255, 0, 0]], dtype=np.uint8)).save(root / 'mask_sam3/camera/test.png')
            Image.fromarray(np.array([[0, 100, 200]], dtype=np.uint8)).save(root / 'semantic_weights/camera/test.png')
            image = str(root / 'images/camera/test.jpg')
            mask = load_mask(image, (3, 1), root, soft_mask_sigma=0)
            semantic = load_semantic_weight(image, (3, 1), root, equipment_weight=1, background_weight=1)
            np.testing.assert_array_equal(mask.numpy(), [[[0, 1, 1]]])
            np.testing.assert_array_equal(semantic.numpy(), [[[0, 1, 1]]])

    def test_cache_eviction_and_materialized_lifetime(self):
        with patch.dict(os.environ, {'CAMERA_CACHE_GIB': str(64 / 1024**3), 'CAMERA_LOAD_WORKERS': '1'}):
            cache = CameraCache([0, 1, 2], 1, SimpleNamespace(source_path='/test'), None, lambda **kw: SimpleNamespace(**kw))
        calls = []
        def prepare(index):
            calls.append(index)
            return dict(original_image=Tensor(), gt_alpha_mask=Tensor(), semantic_weight=None, index=index)
        cache.prepare_one = prepare
        try:
            cache.prefetch([0, 1, 2])
            self.assertEqual(len(cache.pending), 2)
            first = LazyCamera(cache, 0).materialize()
            self.assertIs(cache.get(0), first)
            cache.get(1)
            self.assertLessEqual(cache.bytes, 64)
            self.assertNotIn(0, cache.cache)
            self.assertEqual(first.index, 0)
            cache.get(0)
            self.assertEqual(calls.count(0), 2)
        finally:
            cache.pool.shutdown()

    def test_submission_ships_current_code_not_result_or_credentials(self):
        def inspect(command, cwd, env):
            package = Path(cwd)
            self.assertTrue((package / 'experiment_node.py').exists())
            self.assertTrue((package / 'training_code/utils/training_checkpoint.py').exists())
            self.assertIn('--stop_after', (package / 'training_code/train.py').read_text())
            self.assertTrue((package / 'training_code/experiments/next_16/configs/E09.json').exists())
            self.assertFalse((package / 'training_code/result').exists())
            self.assertFalse((package / '.wandb_auth.json').exists())
            self.assertNotIn('WANDB_API_KEY', env)
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, {'EXPERIMENT_ID': 'E05', 'WANDB_MODE': 'disabled'}), patch('submit_with_auth.subprocess.run', inspect):
            self.assertEqual(submit(['nebulactl', 'run', 'mdl', '--file.cluster_file=cluster.json']), 0)


if __name__ == '__main__':
    unittest.main()
