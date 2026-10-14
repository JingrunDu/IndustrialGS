import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from restore_delivery import merge


class RestoreTests(unittest.TestCase):
    def test_merge_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            chunks = [b'first', b'second']
            parts = []
            for index, chunk in enumerate(chunks):
                name = 'model.part%d' % index
                (directory / name).write_bytes(chunk)
                parts.append(dict(file=name, size=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
            record = directory / 'model.parts.json'
            record.write_text(json.dumps(dict(original='model', original_size=11,
                                             sha256=hashlib.sha256(b'firstsecond').hexdigest(), parts=parts)))
            merge(record)
            self.assertEqual((directory / 'model').read_bytes(), b'firstsecond')
            merge(record)

    def test_bad_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'part').write_bytes(b'wrong')
            record = directory / 'model.parts.json'
            record.write_text(json.dumps(dict(original='model', original_size=5,
                                             sha256='0' * 64, parts=[dict(file='part', size=5, sha256='0' * 64)])))
            with self.assertRaises(ValueError):
                merge(record)
            self.assertFalse((directory / 'model').exists())


if __name__ == '__main__':
    unittest.main()
