"""A checkpoint between gradient-accumulation updates must preserve the next update."""
import random
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import torch
from utils.training_checkpoint import PARAMETERS, load_checkpoint, save_checkpoint


class Model:
    optimizer_type = 'default'

    def __init__(self):
        for name in PARAMETERS:
            setattr(self, name, torch.nn.Parameter(torch.ones(2)))
        self.optimizer = torch.optim.Adam([getattr(self, n) for n in PARAMETERS], lr=0.01)

    def capture(self, optimizer_type):
        return [getattr(self, n) for n in PARAMETERS], self.optimizer.state_dict()

    def restore(self, state, opt):
        for name, value in zip(PARAMETERS, state[0]):
            setattr(self, name, value)
        self.optimizer = torch.optim.Adam([getattr(self, n) for n in PARAMETERS], lr=0.01)
        self.optimizer.load_state_dict(state[1])

    def backward(self):
        sum((getattr(self, n)**2).sum() for n in PARAMETERS).backward()


class CheckpointTests(unittest.TestCase):
    def test_pending_gradients_rng_and_next_optimizer_update(self):
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.get_rng_state_all', return_value=[]), patch('torch.cuda.set_rng_state_all'):
            model = Model()
            model.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            model.backward()  # Halfway through the next accumulated update.
            path = Path(tmp) / 'checkpoint_17.pth'
            save_checkpoint(path, model, 17, [4, 7], .25, torch.zeros(3), 'test')
            expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
            model.backward()
            model.optimizer.step()
            restored = Model()
            state = load_checkpoint(path, restored, None, 'test')
            actual_random = (random.random(), float(np.random.rand()), torch.rand(3))
            self.assertEqual(expected_random[:2], actual_random[:2])
            self.assertTrue(torch.equal(expected_random[2], actual_random[2]))
            self.assertEqual(state['remaining'], [4, 7])
            restored.backward()
            restored.optimizer.step()
            for name in PARAMETERS:
                self.assertTrue(torch.equal(getattr(model, name), getattr(restored, name)))
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_checkpoint(path, Model(), None, 'changed-config')
            with path.open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_checkpoint(path, Model(), None, 'test')


if __name__ == '__main__':
    unittest.main()
