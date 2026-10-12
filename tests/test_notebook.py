"""Notebook contract checks; no notebook actions or training are executed."""
import ast
import json
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/Noise2Inverse360_HPC_v3.3.ipynb'


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.sources = {cell['id']: ''.join(cell['source']) for cell in cls.nb['cells']}

    def test_python38_syntax_and_no_execution_outputs(self):
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), feature_version=(3, 8))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_config_accepts_nafnet_with_seven_or_nine_slices(self):
        parsed = ast.parse(self.sources['training-helpers'])
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'validate_denoise_config')
        module = ast.Module(body=[function], type_ignores=[])
        scope = dict(np=np)
        exec(compile(module, 'notebook_config_validator', 'exec'), scope)
        root = Path('/scratch/run')
        for k in (5, 7, 9):
            cfg = dict(dataset=dict(directory_to_reconstructions=str(root), full_recon_name='full',
                                     sub_recon_name0='even', sub_recon_name1='odd'),
                        train=dict(mode='2.5d', architecture='nafnet25d', precision='bf16',
                                   psz=256, n_slices=k, mbsz=4, maxep=2000, warmup=2000,
                                   patience=100, lr=.001, z_stride=1),
                        infer=dict(window='cosine', overlap=.5))
            scope['validate_denoise_config'](cfg, root, root/'full', root/'even', root/'odd',
                                             2560, nrows=624, volume=True)

    def test_compatibility_guards_and_outputs(self):
        self.assertIn("'--architecture'", self.sources['configure-training'])
        self.assertIn('Saved architecture differs', self.sources['configure-training'])
        for name in ('volume', 'single-slice', 'cleanup-helpers'):
            self.assertIn('_nafnet25d', self.sources[name])
        self.assertIn('RUN_TRAINING = False', self.sources['train'])
        self.assertIn('if not RUN_TRAINING:', self.sources['train'])
        self.assertIn("env['MPLBACKEND'] = 'Agg'", self.sources['core-helpers'])
        self.assertIn("'3.0', '3.1', '3.2', '3.3'", self.sources['core-helpers'])
        self.assertIn('zoom_image_viewer', self.sources['single-slice'])
        self.assertIn('plot_training_history', self.sources['train'])


if __name__ == '__main__':
    unittest.main()
