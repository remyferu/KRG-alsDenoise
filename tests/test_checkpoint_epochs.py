"""Checkpoint epoch provenance without training or optimizer updates."""
import contextlib
import ast
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'workflow/src'))
from denoise.checkpoints import CHECKPOINT_NAMES, load_inference_model, save_checkpoint
from denoise.model_factory import build_model, model_spec
from denoise.tiffs import save_stack
from krg_workflow.denoising import inference_checkpoint_info


class CheckpointEpochTests(unittest.TestCase):
    def test_raw_best_checkpoint_writers_save_the_current_epoch(self):
        tree = ast.parse((ROOT/'denoise/train.py').read_text(encoding='utf-8'))
        conditions = {"val_values['lcl'] < best_lcl_loss", "val_values['l1'] < best_val_loss",
                      "val_values['edge'] > best_edge"}
        branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) in conditions]
        self.assertEqual(len(branches), 3)
        with tempfile.TemporaryDirectory() as temporary:
            scope = dict(val_values=dict(l1=1.,lcl=.1,edge=2.), best_val_loss=float('inf'),
                         best_lcl_loss=float('inf'),best_edge=0.,epoch=37,odir=temporary,
                         spec=dict(version=2),deepcopy=deepcopy,save_checkpoint=save_checkpoint,
                         model=SimpleNamespace(module=SimpleNamespace(state_dict=lambda: dict(weight=torch.ones(1)))),
                         optimizer=SimpleNamespace(state_dict=lambda: {}))
            for branch in branches:
                exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                             'checkpoint-selection','exec'),scope)
            for kind in ('val','lcl','edge'):
                saved = torch.load(Path(temporary)/CHECKPOINT_NAMES[kind],weights_only=False)
                self.assertEqual(saved['epoch'],37)

    def test_all_checkpoint_selections_report_their_saved_epoch(self):
        params = dict(train=dict(architecture='nafnet25d', n_slices=3,
                                 nafnet=dict(width=4, stem_width=2, detail_blocks=0,
                                             encoder_blocks=[0, 0], middle_blocks=0, decoder_blocks=[0, 0])),
                      dataset=dict(directory_to_reconstructions='unused'))
        model = build_model(params)
        with tempfile.TemporaryDirectory() as temporary:
            for index, (kind, filename) in enumerate(CHECKPOINT_NAMES.items(), 12):
                save_checkpoint(dict(model_spec=model_spec(params), model_state_dict=model.state_dict(), epoch=index),
                                Path(temporary)/filename)
                with patch('denoise.checkpoints.log.info') as logger:
                    loaded = load_inference_model(params, SimpleNamespace(checkpoint=kind, model_dir=temporary), '2.5d', 'cpu')
                self.assertEqual(loaded.inference_checkpoint['epoch'], index)
                self.assertEqual(loaded.inference_checkpoint['filename'], filename)
                self.assertFalse(loaded.training)
                self.assertIn('training epoch: '+str(index), logger.call_args[0][0])
            save_checkpoint(dict(model_spec=model_spec(params), model_state_dict=model.state_dict()),
                            Path(temporary)/CHECKPOINT_NAMES['val'])
            save_checkpoint(dict(epoch=999), Path(temporary)/'resume.pth')
            with patch('denoise.checkpoints.log.info') as logger:
                loaded = load_inference_model(params, SimpleNamespace(checkpoint='val', model_dir=temporary), '2.5d', 'cpu')
            self.assertIsNone(loaded.inference_checkpoint['epoch'])
            self.assertIn('unknown (not recorded)', logger.call_args[0][0])

    def test_cached_tiffs_keep_the_epoch_used_even_if_the_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root/'best_ema_model.pth'
            info = dict(filename=weights.name, path=str(weights), epoch=37)
            with contextlib.redirect_stdout(io.StringIO()):
                save_stack(root/'volume', np.ones((2, 8, 8), dtype=np.float32), offset=5,
                           metadata=dict(inference_checkpoint=info))
            save_checkpoint(dict(epoch=99), weights)
            for row in (5, 6):
                path = root/'volume'/('{:05d}.tiff'.format(row))
                self.assertEqual(inference_checkpoint_info(path), info)
                np.testing.assert_array_equal(tifffile.imread(path), np.ones((8, 8)))

    def test_old_or_unrecognized_tiff_metadata_reports_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'old.tiff'
            for options in ({}, dict(description='legacy denoised image'),
                            dict(metadata=dict(inference_checkpoint=dict(epoch=True)))):
                tifffile.imwrite(path, np.ones((8, 8), dtype=np.float32), **options)
                self.assertIsNone(inference_checkpoint_info(path)['epoch'])


if __name__ == '__main__':
    unittest.main()
