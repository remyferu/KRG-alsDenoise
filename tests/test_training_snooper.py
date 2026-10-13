"""Concurrent log snapshots and observer isolation; no model execution."""
import ast
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'workflow/src'))
from krg_workflow import history


def epoch_record(epoch):
    return (f'Epoch {epoch}\n[Train] L1 loss: 1.1\n[Train] LCL loss: .01\n'
            '[Val] L1 loss: 1.2\n[Val] LCL loss: .02\n'
            '[Train] Total loss: .555\n[Val] Total loss: .61\n'
            '[Train] L2 loss: .3\n[Val] L2 loss: .4\n'
            '[Train] SSIM loss: .1\n[Val] SSIM loss: .2\n')


class SnooperTests(unittest.TestCase):
    def test_missing_or_incomplete_log_creates_no_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'train.log', root/'observer'
            result = history.snapshot_training_history(source, output_root=output, show=False)
            self.assertEqual(result['status'], 'no_log')
            self.assertFalse(output.exists())
            for text in ('', 'Epoch 1\n[Train] L1 loss: 1.2\n', 'Epoch 1'):
                source.write_text(text, encoding='utf-8')
                before = source.read_bytes()
                result = history.snapshot_training_history(source, output_root=output, show=False)
                self.assertEqual(result['status'], 'no_history')
                self.assertEqual(source.read_bytes(), before)
                self.assertFalse(output.exists())

    def test_snapshot_excludes_partial_tail_and_leaves_training_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root/'logs'
            logs.mkdir()
            source = logs/'train.log'
            source.write_bytes((epoch_record(1)+'Epoch 2\n[Train] L1 loss: 1\n'
                                '[Train] LCL loss: .01\n[Val] L1 loss: 1\n[Val] LCL loss: .02').encode())
            lock = root/'.run.lock'
            lock.write_text('active training lock', encoding='utf-8')
            training_exports = root/'training_history'
            training_exports.mkdir()
            existing = training_exports/'training_loss_history.png'
            existing.write_bytes(b'training-owned export')
            before = source.read_bytes()
            modified = source.stat().st_mtime_ns
            result = history.snapshot_training_history(source, show=False)
            self.assertEqual(result['status'], 'saved')
            self.assertEqual(result['last_epoch'], 1)
            self.assertGreater(result['omitted_tail_bytes'], 0)
            self.assertEqual(result['log_tail'][-1], '[Val] L1 loss: 1')
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(source.stat().st_mtime_ns, modified)
            self.assertEqual(lock.read_text(), 'active training lock')
            self.assertEqual(existing.read_bytes(), b'training-owned export')
            for path in result['paths'].values():
                self.assertTrue(Path(path).is_file())
                self.assertTrue(str(Path(path)).startswith(str(logs/'training_snooper')))
            saved = json.loads(Path(result['paths']['snapshot']).read_text(encoding='utf-8'))
            self.assertEqual(saved['log_path'], str(source.resolve()))
            self.assertEqual(saved['captured_bytes'], len(before)-result['omitted_tail_bytes'])
            with Path(result['paths']['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['epoch'] for r in rows], ['1'])
            self.assertEqual(rows[0]['eval_total'], '0.61')

    def test_append_during_plot_does_not_change_captured_view_and_refresh_is_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'train.log'
            output = root/'observer'
            source.write_text(epoch_record(1), encoding='utf-8')
            original_plot = history.plot_training_history
            observed = []

            def append_while_plotting(path, *args, **kwargs):
                with source.open('a', encoding='utf-8') as stream:
                    stream.write(epoch_record(2))
                observed.append(Path(path).read_text(encoding='utf-8'))
                return original_plot(path, *args, **kwargs)

            with patch.object(history, 'plot_training_history', side_effect=append_while_plotting):
                first = history.snapshot_training_history(source, output_root=output, show=False)
            self.assertEqual(first['last_epoch'], 1)
            self.assertNotIn('Epoch 2', observed[0])
            self.assertIn('Epoch 2', source.read_text())
            first_png = Path(first['paths']['png']).read_bytes()
            second = history.snapshot_training_history(source, output_root=output, show=False)
            self.assertEqual(second['last_epoch'], 2)
            self.assertNotEqual(first['paths']['png'], second['paths']['png'])
            self.assertEqual(Path(first['paths']['png']).read_bytes(), first_png)

    def test_ansi_tail_is_readable_and_parser_handles_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'train.log'
            source.write_text('\x1b[92m'+epoch_record(1)+'\x1b[0m\n'
                              'denoise train --resume --gpus 0,1\n'+epoch_record(1)+epoch_record(2), encoding='utf-8')
            result = history.snapshot_training_history(source, show=False)
            self.assertEqual(result['epochs'], 2)
            self.assertTrue(all('\x1b' not in line for line in result['log_tail']))

    def test_import_does_not_load_torch(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/'workflow/src'), MPLBACKEND='Agg', PYTHONNOUSERSITE='1')
        command = [sys.executable, '-c',
                   "import sys; from krg_workflow.history import snapshot_training_history; assert 'torch' not in sys.modules"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_notebook_is_independent_and_has_no_execution_outputs(self):
        notebook = json.loads((ROOT/'notebooks/Training_Snooper.ipynb').read_text(encoding='utf-8'))
        code = []
        titles = []
        for cell in notebook['cells']:
            text = ''.join(cell['source'])
            if cell['cell_type'] == 'code':
                ast.parse(text, feature_version=(3, 8))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                titles.append(int(re.match(r'# (\d+)\.', text)[1]))
                code.append(text)
        self.assertEqual(titles, [1, 2, 3, 4])
        combined = '\n'.join(code)
        self.assertIn('snapshot_training_history', combined)
        self.assertNotIn('ct.', combined)
        self.assertNotIn('import torch', combined)
        self.assertNotIn('run_logged', combined)
        self.assertNotIn('run.', combined)


if __name__ == '__main__':
    unittest.main()
