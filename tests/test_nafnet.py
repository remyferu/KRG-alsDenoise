import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tifffile
import torch
import yaml

from denoise.nafnet25d import NAFNet25D, LayerNorm2d
from denoise.model_factory import build_model, model_spec, load_model, forward_model
from denoise.data import TomoDatasetTrain, TomoDatasetInfer
from denoise.data_utils import (InferenceBatchSizeOptimizer,
                               extract_sliding_window_patches_25d,
                               stitch_sliding_window_patches)
from examples.configure_nafnet import configure


def config(n_slices=5):
    return dict(train=dict(mode='2.5d', architecture='nafnet25d', n_slices=n_slices,
                           psz=16, nafnet=dict(width=8, stem_width=4, detail_blocks=1,
                                              encoder_blocks=[1, 1], middle_blocks=1,
                                              decoder_blocks=[1, 1])),
                infer=dict(overlap=0.5, window='cosine', num_workers=0, max_batch_size=2))


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_and_odd_shapes_for_all_slice_counts(self):
        for k in (5, 7, 9):
            for h, w in ((16, 16), (17, 19), (1, 2)):
                with self.subTest(k=k, size=(h, w)):
                    model = build_model(config(k))
                    x = torch.randn(2, k, h, w)
                    torch.testing.assert_close(model(x), x[:, k//2:k//2+1], rtol=0, atol=0)

    def test_invalid_input_and_options(self):
        with self.assertRaises(ValueError):
            NAFNet25D(n_slices=6)
        with self.assertRaises(ValueError):
            build_model(config())(torch.randn(1, 7, 16, 16))
        bad = config(); bad['train']['nafnet']['widht'] = 32
        with self.assertRaises(ValueError):
            build_model(bad)
        with self.assertRaises(ValueError):
            build_model(config(), '3d')

    def test_layer_norm_matches_pixel_channel_normalization(self):
        x = torch.randn(2, 8, 13, 17)
        mean = x.mean(1, keepdim=True)
        variance = ((x - mean) ** 2).mean(1, keepdim=True)
        expected = (x - mean) / torch.sqrt(variance + 1e-6)
        torch.testing.assert_close(LayerNorm2d(8)(x), expected)

    @unittest.skipUnless(os.environ.get('RUN_TRAINING_TESTS') == '1', 'Training tests are opt-in')
    def test_learns_and_has_finite_gradients_in_every_branch(self):
        torch.manual_seed(41)
        model = build_model(config())
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        x, target = torch.randn(2, 5, 16, 16), torch.zeros(2, 1, 16, 16)
        initial = torch.nn.functional.l1_loss(model(x), target).item()
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x), target)
            loss.backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            opt.step()
        self.assertLess(loss.item(), initial)
        for branch in ('stem', 'neighbor_gate', 'detail', 'middle', 'dec0'):
            self.assertGreater(sum(p.grad.abs().sum().item() for name, p in model.named_parameters()
                                   if name.startswith(branch)), 0, branch)

    def test_checkpoint_roundtrip_and_mismatch(self):
        params = config()
        model = build_model(params)
        ckpt = dict(model_spec=model_spec(params), model_state_dict=model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pth'
            torch.save(ckpt, path)
            restored = load_model(params, torch.load(path, weights_only=False))
            x = torch.randn(1, 5, 16, 16)
            torch.testing.assert_close(model(x), restored(x))
        for bad in (config(7), {'train': {'mode': '2.5d', 'n_slices': 5}}):
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_model(bad, ckpt)
        ckpt.pop('model_spec')
        with self.assertRaisesRegex(ValueError, 'fresh NAFNet'):
            load_model(params, ckpt)

    def test_legacy_models_and_checkpoints(self):
        for train in (dict(n_slices=5), dict(mode='3d', n_blocks_3d=2, start_filts_3d=8)):
            params = dict(train=train)
            original = build_model(params)
            restored = load_model(params, {'model_state_dict': original.state_dict()})
            x = torch.randn(1, 5, 16, 16) if 'mode' not in train else torch.randn(1, 1, 16, 16, 16)
            torch.testing.assert_close(original(x), restored(x))

    @unittest.skipUnless(os.environ.get('RUN_TRAINING_TESTS') == '1' and torch.cuda.is_available(),
                         'CUDA training tests are opt-in')
    def test_cuda_two_view_fp32_and_bf16_optimizer_steps(self):
        from denoise.loss import LCL
        for precision in ('fp32', 'bf16'):
            if precision == 'bf16' and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(precision=precision):
                model = build_model(config()).cuda()
                opt = torch.optim.Adam(model.parameters())
                x, y = torch.randn(2, 5, 32, 32, device='cuda'), torch.randn(2, 5, 32, 32, device='cuda')
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    a, b = forward_model(model, x, precision), forward_model(model, y, precision)
                    loss = .5 * ((a-y[:, 2:3]).abs().mean() + (b-x[:, 2:3]).abs().mean())
                    loss = loss + .005 * (LCL()(a) + LCL()(b))
                    self.assertEqual(a.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward(); opt.step()
                del model, opt, a, b, loss
                torch.cuda.empty_cache()


class WorkflowTests(unittest.TestCase):
    def test_crop_preserves_pairing_at_boundaries_and_exact_image_size(self):
        ds = object.__new__(TomoDatasetTrain)
        ds.n_slices = 7; ds.psz = 8
        ds.split0 = np.arange(3*8*8, dtype=np.float32).reshape(3, 8, 8)
        ds.split1 = ds.split0 + 1000
        ds.augmentations = lambda image, split1: dict(image=image, split1=split1)
        a, b = ds[0]
        np.testing.assert_array_equal(a, ds.split0[[0, 0, 0, 0, 1, 2, 2]])
        np.testing.assert_array_equal(b-a, np.full(a.shape, 1000))

    def test_stitching_identity_including_cosine_window_edges(self):
        x = torch.randn(1, 7, 23, 25)
        patches, coords, meta = extract_sliding_window_patches_25d(x, (16, 16), overlap=.5)
        for window in ('uniform', 'hann', 'cosine'):
            result = stitch_sliding_window_patches(patches[:, :, 3:4], coords, meta, window=window)
            torch.testing.assert_close(result, x[:, 3:4])

    def test_cpu_profiler_uses_requested_channels_and_propagates_errors(self):
        profiler = InferenceBatchSizeOptimizer(build_model(config(9)), (16, 16),
                                               torch.device('cpu'), n_channels=9)
        self.assertEqual(profiler.profile()['optimal_batch_size'], 1)
        wrong = InferenceBatchSizeOptimizer(build_model(config(9)), (16, 16),
                                            torch.device('cpu'), n_channels=5)
        with self.assertRaisesRegex(ValueError, 'Expected'):
            wrong.profile()

    def test_config_tool_preserves_dataset_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'old.yaml', Path(tmp)/'naf.yaml'
            original = dict(dataset=dict(full_recon_name='scan_rec', mean4norm=2, std4norm=3),
                            train=dict(mode='3d', n_slices=5))
            src.write_text(yaml.safe_dump(original))
            result = configure(src, dst, n_slices=9)
            self.assertEqual(result['dataset'], original['dataset'])
            self.assertEqual(yaml.safe_load(src.read_text()), original)
            self.assertEqual(result['train']['architecture'], 'nafnet25d')
            with self.assertRaises(ValueError):
                configure(src, src)
            with self.assertRaises(ValueError):
                configure(src, dst)

    def test_real_tiff_slice_and_volume_inference_for_seven_slices(self):
        from denoise import slice as slice_cmd, volume as volume_cmd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images_dir = root/'scan_rec'; images_dir.mkdir()
            volume = np.random.default_rng(1).normal(2, .5, (3, 20, 21)).astype(np.float32)
            for i, image in enumerate(volume):
                tifffile.imwrite(images_dir/('%05d.tiff' % i), image)
            params = config(7)
            params['dataset'] = dict(directory_to_reconstructions=tmp, full_recon_name='scan_rec',
                                     mean4norm=2., std4norm=.5)
            cfg_path = root/'config.yaml'; cfg_path.write_text(yaml.safe_dump(params))
            model_dir = root/'models'; model_dir.mkdir()
            model = build_model(params)
            torch.save(dict(model_spec=model_spec(params), model_state_dict=model.state_dict()),
                       model_dir/'best_val_model.pth')
            args = SimpleNamespace(config=str(cfg_path), checkpoint='val', model_dir=str(model_dir),
                                    slice_number=1, start_slice='', end_slice='', mode=None)
            with patch('torch.cuda.is_available', return_value=False):
                slice_cmd.run(args)
                volume_cmd.run(args)
            actual = tifffile.imread(root/'scan_denoised_slices_nafnet25d'/'00001.tiff')
            np.testing.assert_allclose(actual, volume[1], atol=1e-5)
            output = sorted((root/'scan_denoised_volume_2.5d_nafnet25d').glob('*.tif*'))
            self.assertEqual(len(output), 3)
            np.testing.assert_allclose(np.stack([tifffile.imread(p) for p in output]), volume, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
