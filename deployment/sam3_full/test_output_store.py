import io
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

from output_store import ImageWriter, LocalStore


class OutputTests(unittest.TestCase):
    def test_complete_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            store = LocalStore(temp)
            writer = ImageWriter(store, workers=2, capacity=2)
            a = np.zeros((8, 8), np.uint8)
            try:
                writer.submit('cam/a.jpg', {'mask_sam3': a, 'semantic_weights': a + 200})
                completed = []
                while writer.futures:
                    completed.extend(writer.collect(timeout=1))
                self.assertEqual(len(completed), 1)
                path = Path(temp) / 'mask_sam3/cam/a.png'
                before = path.read_bytes()
                writer.submit('cam/a.jpg', {'mask_sam3': a + 255})
                while writer.futures:
                    writer.collect(timeout=1)
                self.assertEqual(before, path.read_bytes())
                self.assertEqual(store.existing('semantic_weights'), {'cam/a.png'})
                with Image.open(io.BytesIO(before)) as image:
                    image.verify()
            finally:
                writer.close()

    def test_bounded_queue_and_failed_pair_not_complete(self):
        gate = threading.Event()

        class BrokenStore:
            def put(self, relative, payload):
                gate.wait(timeout=5)
                if relative.startswith('semantic_weights/'):
                    raise RuntimeError('simulated upload failure')

        writer = ImageWriter(BrokenStore(), workers=1, capacity=1)
        try:
            a = np.zeros((8, 8), np.uint8)
            writer.submit('a.jpg', {'mask_sam3': a, 'semantic_weights': a})
            self.assertEqual(writer.collect(), [])
            with self.assertRaisesRegex(RuntimeError, 'queue full'):
                writer.submit('b.jpg', {'mask_sam3': a})
            gate.set()
            with self.assertRaisesRegex(RuntimeError, 'simulated upload failure'):
                while writer.futures:
                    writer.collect(timeout=1)
        finally:
            gate.set()
            writer.close()


if __name__ == '__main__':
    unittest.main()
