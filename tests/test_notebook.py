"""Notebook contract checks; no notebook actions or training are executed."""
import ast
import json
import unittest
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/Noise2Inverse360_HPC_v3.4.ipynb'
sys.path.insert(0,str(ROOT/'workflow/src'))


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
        from krg_workflow.denoising import validate_denoise_config
        root = Path('/scratch/run')
        for k in (5, 7, 9):
            cfg = dict(dataset=dict(directory_to_reconstructions=str(root), full_recon_name='full',
                                     sub_recon_name0='even', sub_recon_name1='odd'),
                        train=dict(mode='2.5d', architecture='nafnet25d', precision='bf16',
                                   psz=256, n_slices=k, mbsz=4, maxep=2000, warmup=2000,
                                   patience=100, lr=.001, z_stride=1),
                        infer=dict(window='cosine', overlap=.5))
            validate_denoise_config(cfg, root, root/'full', root/'even', root/'odd',
                                             2560, nrows=624, volume=True)

    def test_short_actions_and_training_opt_in(self):
        self.assertEqual(len([c for c in self.nb['cells'] if c['cell_type']=='code']),20)
        code = '\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        self.assertNotIn('globals()',code)
        self.assertNotIn('def ',code)
        self.assertIn('RUN_TRAINING = False',self.sources['train'])
        self.assertIn('enabled=RUN_TRAINING',self.sources['train'])
        self.assertIn('live_log_box()',self.sources['train'])
        self.assertIn('krg_workflow.__version__',self.sources['imports'])
        for path in (ROOT/'workflow/src/krg_workflow').glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))

    def test_setup_includes_workflow_and_als_installation(self):
        setup = self.sources['installation-and-workspace']
        self.assertIn('conda env create',setup)
        self.assertIn('KRG-alsDenoise/workflow',setup)
        self.assertIn('ALS CT + N2I notebook',setup)
        self.assertIn('python=3.11',setup)


if __name__ == '__main__':
    unittest.main()
