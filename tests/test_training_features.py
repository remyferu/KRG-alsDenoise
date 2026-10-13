"""v3.5 policy, data and inference checks; no backward pass or optimizer updates."""
from copy import deepcopy
from datetime import timedelta
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from denoise.data import TomoDatasetTrain
from denoise.early_stopping import MovingAverageEarlyStopping
from denoise.evaluation import evaluate_pairs
from denoise.loss import LCL
from denoise.model_factory import model_spec, load_model
from denoise.nafnet25d import NAFNet25D
from denoise.reference_metrics import SSIMLoss, resolve_ssim_data_range
from denoise.training_controls import (ModelEMA, PlateauController, plateau_options,
                                       ema_options, update_training_controls)
from denoise.validation import (validation_options, validation_plan, FixedValidationDataset,
                                EvaluationShard, EpochMetrics)
from denoise.checkpoints import initial_checkpoint, load_inference_model, save_checkpoint


class CenterIdentity(nn.Module):
    def forward(self, x):
        return x[:, x.shape[1]//2:x.shape[1]//2+1]


class SyntheticPairs(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        random = torch.Generator().manual_seed(index)
        x = torch.randn(5, 16, 16, generator=random)
        return x, x+2


def distributed_evaluation_worker(rank, world_size, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    torch.distributed.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
        rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        dataset = SyntheticPairs()
        loader = DataLoader(dataset, batch_size=1, sampler=EvaluationShard(dataset, rank, world_size))
        values, count, preview = evaluate_pairs(CenterIdentity(), loader, '2.5d', 'fp32', 'cpu',
            nn.L1Loss(), LCL(), .01, True, SSIMLoss(10))
        ema, ema_count, _ = evaluate_pairs(CenterIdentity(), loader, '2.5d', 'fp32', 'cpu',
            nn.L1Loss(), LCL(), .01, True, SSIMLoss(10), reference=False)
        (root/f'rank{rank}.json').write_text(json.dumps(dict(values=values, count=count,
            ema=ema, ema_count=ema_count, empty=preview is None)), encoding='utf-8')
    finally:
        torch.distributed.destroy_process_group()


class FixedValidationTests(unittest.TestCase):
    def test_unique_reproducible_crops_and_disjoint_input_planes(self):
        options = validation_options({})
        for k in (5, 7, 9):
            plan = validation_plan((64, 32, 34), k, 16, options)
            self.assertEqual(plan, validation_plan((64, 32, 34), k, 16, options))
            self.assertEqual(len(plan['coordinates']), len(set(map(tuple, plan['coordinates']))))
            train_planes = {min(63, max(0, center+offset))
                for center in range(plan['training_centers'][1]) for offset in range(-k//2+1, k//2+1)}
            val_planes = {min(63, max(0, center+offset))
                for center, _, _ in plan['coordinates'] for offset in range(-k//2+1, k//2+1)}
            self.assertFalse(train_planes & val_planes)
            self.assertEqual(max(train_planes)+1, plan['normalization_planes'][1])

    def test_small_geometry_caps_samples_and_shards_without_duplicates(self):
        plan = validation_plan((12, 16, 16), 5, 16, validation_options({}))
        self.assertEqual(len(plan['coordinates']), 3)
        shards = [list(EvaluationShard(plan['coordinates'], rank, 5)) for rank in range(5)]
        self.assertEqual(sorted(index for shard in shards for index in shard), [0, 1, 2])
        self.assertEqual(shards[-1], [])

    def test_training_only_normalization_and_unaugmented_fixed_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (32, 16, 16)
            original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)/100
            raw = original.copy()
            raw[27:] += 1000
            for folder, volume in (('even', raw), ('odd', 2*raw+4)):
                (root/folder).mkdir()
                for index, image in enumerate(volume):
                    tifffile.imwrite(root/folder/f'{index:05d}.tiff', image)
            params = dict(dataset=dict(directory_to_reconstructions=str(root),
                sub_recon_name0='even', sub_recon_name1='odd'),
                train=dict(psz=16, n_slices=5, validation=dict(samples=8)))
            dataset = TomoDatasetTrain(params, root/'config.yaml')
            np.testing.assert_allclose(dataset.split0_mean, original[:27].mean(dtype=np.float64), rtol=1e-7)
            np.testing.assert_allclose(dataset.split0_std, original[:27].std(dtype=np.float64), rtol=1e-7)
            fixed = FixedValidationDataset(dataset)
            with patch.object(dataset, 'augmentations', side_effect=AssertionError('Validation augmentation')):
                first = fixed[0]
                np.testing.assert_array_equal(first[0], fixed[0][0])
                np.testing.assert_allclose(first[0], first[1], rtol=2e-6)
            span = resolve_ssim_data_range({}, dataset)
            expected = max(dataset.split0[:27].max(), dataset.split1[:27].max())-min(
                dataset.split0[:27].min(), dataset.split1[:27].min())
            self.assertAlmostEqual(span, float(expected), places=6)
            self.assertLess(span, 10)

    def test_invalid_geometry_and_policy(self):
        for config in ({'validation': None}, {'validation': {'fraction': .5}},
                       {'validation': {'samples': 0}}, {'validation': {'seed': -1}},
                       {'validation': {'unused': 1}}):
            with self.assertRaises(ValueError):
                validation_options(config)
        with self.assertRaises(ValueError):
            validation_options({'validation': {'enabled': True}}, '3d')
        with self.assertRaises(ValueError):
            validation_plan((5, 16, 16), 5, 16, validation_options({}))


class SchedulerTests(unittest.TestCase):
    def controls(self):
        optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(()))], lr=.001)
        options = plateau_options(dict(lr=.001, lr_schedule=dict(patience=1, cooldown=0,
                                     min_lr=.00025, refinement_epochs=3)))
        return optimizer, PlateauController(optimizer, options, .001, window=1), MovingAverageEarlyStopping(
            patience=2, window=1, min_delta=.001)

    def test_reductions_floor_and_refinement_do_not_reset_best_or_patience(self):
        optimizer, controller, stopping = self.controls()
        decisions = []
        for epoch in range(1, 9):
            decisions.append(update_training_controls(stopping, controller, epoch, 1., False))
        self.assertEqual([index+1 for index, (_, reduced) in enumerate(decisions) if reduced], [3, 5])
        self.assertEqual(optimizer.param_groups[0]['lr'], .00025)
        self.assertEqual(stopping.best_mean, 1.)
        self.assertEqual(stopping.epochs_since_improvement, 7)
        self.assertFalse(any(stop for stop, _ in decisions[:7]))
        self.assertTrue(decisions[7][0])

    def test_complete_window_after_warmup_and_resume_are_exact(self):
        optimizer, controller, stopping = self.controls()
        stopping = MovingAverageEarlyStopping(patience=100, window=2)
        controller = PlateauController(optimizer, controller.options, stopping.min_delta, stopping.window)
        for epoch, warm in ((1, True), (2, True), (3, False)):
            update_training_controls(stopping, controller, epoch, 1., warm)
            self.assertEqual(controller.scheduler.best, float('inf'))
        update_training_controls(stopping, controller, 4, 1., False)
        self.assertEqual(controller.scheduler.best, 1.)
        saved = dict(optimizer=deepcopy(optimizer.state_dict()), scheduler=deepcopy(controller.state_dict()),
                     stopping=deepcopy(stopping.state_dict()))
        resumed_optimizer, _, _ = self.controls()
        resumed_optimizer.load_state_dict(saved['optimizer'])
        resumed = PlateauController(resumed_optimizer, controller.options, stopping.min_delta, stopping.window)
        self.assertTrue(resumed.load_state_dict(saved['scheduler']))
        monitor = MovingAverageEarlyStopping(patience=100, window=2)
        self.assertTrue(monitor.load_state_dict(saved['stopping']))
        for epoch in range(5, 12):
            expected = update_training_controls(stopping, controller, epoch, 1., False)
            self.assertEqual(update_training_controls(monitor, resumed, epoch, 1., False), expected)
            self.assertEqual(resumed_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
        changed = PlateauController(resumed_optimizer, controller.options, stopping.min_delta, window=3)
        self.assertFalse(changed.load_state_dict(saved['scheduler']))

    def test_invalid_options(self):
        for options in ({'factor': 1}, {'factor': float('nan')}, {'patience': -1},
                        {'min_lr': .001}, {'refinement_epochs': -1}, {'unexpected': 1}):
            with self.assertRaises(ValueError):
                plateau_options(dict(lr=.001, lr_schedule=options))
        for options in ({'decay': 1}, {'decay': float('nan')}, {'evaluate_every': 0}, {'unexpected': 1}):
            with self.assertRaises(ValueError):
                ema_options(dict(ema=options))


class EMATests(unittest.TestCase):
    def test_failed_checkpoint_write_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'resume.pth'
            save_checkpoint(dict(epoch=1), path)
            original = path.read_bytes()
            def failed_save(state, target):
                Path(target).write_bytes(b'incomplete checkpoint')
                raise OSError('Interrupted write')
            with patch('denoise.checkpoints.torch.save', side_effect=failed_save):
                with self.assertRaises(OSError):
                    save_checkpoint(dict(epoch=2), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_seed_decay_buffers_and_resume_without_training(self):
        raw = nn.Linear(1, 1, bias=False)
        raw.register_buffer('counter', torch.tensor(0))
        average = ModelEMA(raw, .5)
        with torch.no_grad():
            raw.weight.fill_(2)
        average.update(raw, 1, 2)
        self.assertEqual(average.num_updates, 0)
        with torch.no_grad():
            raw.weight.fill_(6)
        average.update(raw, 3, 2)
        with torch.no_grad():
            raw.weight.fill_(10)
            raw.counter.fill_(7)
        average.update(raw, 4, 2)
        self.assertEqual(average.model.weight.item(), 8)
        self.assertEqual(average.model.counter.item(), 7)
        self.assertFalse(average.model.training)
        self.assertFalse(average.model.weight.requires_grad)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'ema.pth'
            torch.save(average.state_dict(), path)
            resumed = ModelEMA(raw, .5)
            self.assertTrue(resumed.load_state_dict(torch.load(path, weights_only=False)))
            resumed.update(raw, 5, 2)
            self.assertEqual(resumed.model.weight.item(), 9)
            self.assertEqual(resumed.num_updates, 3)
        self.assertFalse(ModelEMA(raw, .9).load_state_dict(average.state_dict()))

    def test_ema_checkpoint_is_selectable_for_inference(self):
        params = dict(train=dict(mode='2.5d', architecture='nafnet25d', n_slices=5,
            nafnet=dict(width=8, stem_width=4, detail_blocks=1, encoder_blocks=[1, 1],
                        middle_blocks=1, decoder_blocks=[1, 1])),
            dataset=dict(directory_to_reconstructions='unused'))
        model = NAFNet25D(n_slices=5, **params['train']['nafnet'])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'best_ema_model.pth'
            torch.save(dict(model_spec=model_spec(params), model_state_dict=model.state_dict()), checkpoint)
            loaded = load_inference_model(params, SimpleNamespace(checkpoint='ema', model_dir=temporary),
                                           '2.5d', 'cpu')
            self.assertFalse(loaded.training)
            self.assertEqual(loaded.distance_bias.shape, (2, 4))


class DistanceFusionTests(unittest.TestCase):
    def test_distance_changes_the_actual_fusion_correction(self):
        model = NAFNet25D(width=8, stem_width=4, detail_blocks=1,
            encoder_blocks=[1, 1], middle_blocks=1, decoder_blocks=[1, 1])
        with torch.no_grad():
            for parameter in model.neighbor_gate.parameters():
                parameter.zero_()
        corrections = []
        handle = model.fuse.register_forward_pre_hook(
            lambda layer, inputs: corrections.append(inputs[0][:, 4:].clone()))
        near, far = torch.zeros(1, 5, 8, 8), torch.zeros(1, 5, 8, 8)
        near[:, 1], far[:, 0] = 1, 1
        with torch.inference_mode():
            model(near)
            model(far)
        handle.remove()
        self.assertGreater(corrections[0].abs().sum().item(), 0)
        torch.testing.assert_close(corrections[1], (2/3)*corrections[0], atol=1e-7, rtol=1e-5)

    def test_distance_prior_and_identity_for_five_seven_nine_slices(self):
        for k in (5, 7, 9):
            model = NAFNet25D(n_slices=k, width=8, stem_width=4, detail_blocks=1,
                             encoder_blocks=[1, 1], middle_blocks=1, decoder_blocks=[1, 1])
            expected = -torch.arange(1, k//2+1).float().log()[:, None].expand(-1, 4)
            torch.testing.assert_close(model.distance_bias, expected)
            x = torch.randn(1, k, 17, 19)
            with torch.inference_mode():
                torch.testing.assert_close(model(x), x[:, k//2:k//2+1], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in NAFNet25D().parameters()), 989777)

    def test_old_format_and_distance_ablation_mismatch_are_rejected(self):
        params = dict(train=dict(architecture='nafnet25d', n_slices=5))
        model = NAFNet25D()
        checkpoint = dict(model_spec=model_spec(params), model_state_dict=model.state_dict())
        old = deepcopy(checkpoint)
        old['model_spec']['version'] = 1
        old['model_spec']['options'].pop('distance_aware')
        for saved in (old, dict(checkpoint, model_spec=dict(checkpoint['model_spec'],
                options=dict(checkpoint['model_spec']['options'], distance_aware=False)))):
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_model(params, saved)
        with tempfile.TemporaryDirectory() as temporary:
            torch.save(checkpoint, Path(temporary)/'resume.pth')
            with self.assertRaisesRegex(ValueError, 'v3.5'):
                initial_checkpoint(params, SimpleNamespace(resume=True), '2.5d', temporary)


class MetricTests(unittest.TestCase):
    def test_separate_prepare_config_retains_dataset_paths_and_refuses_overwrite(self):
        from denoise.__main__ import make_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/'scan_rec'
            yaml_path = root/'scan_v3_5_config.yaml'
            args = SimpleNamespace(file_name=str(root/'scan.h5'), out_path_name=str(output),
                                   config_output=str(yaml_path))
            with patch.dict('sys.modules', {'meta': None}):
                make_config(args)
                with self.assertRaises(FileExistsError):
                    make_config(args)
            config = yaml.safe_load(yaml_path.read_text())
            self.assertEqual(config['dataset']['full_recon_name'], 'scan_rec')
            self.assertEqual(config['dataset']['sub_recon_name0'], 'scan_rec_0')
            self.assertFalse((root/'scan_rec_config.yaml').exists())

    def test_sample_weighting_and_collective_sum_with_an_empty_shard(self):
        a, b = EpochMetrics(('total',), 'cpu'), EpochMetrics(('total',), 'cpu')
        a.add(3, total=torch.tensor(2., requires_grad=True))
        b.add(1, total=10.)
        merged = a.totals+b.totals
        empty = EpochMetrics(('total',), 'cpu')
        def reduce(tensor, op):
            tensor.copy_(merged)
        for metrics in (a, b, empty):
            with patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.all_reduce', side_effect=reduce):
                values, count = metrics.reduce()
            self.assertEqual((values['total'], count), (4., 4))
            self.assertFalse(metrics.totals.requires_grad)

    def test_evaluation_objective_and_ema_use_same_pairs(self):
        loader = DataLoader(SyntheticPairs(), batch_size=2)
        raw, count, preview = evaluate_pairs(CenterIdentity(), loader, '2.5d', 'fp32', 'cpu',
            nn.L1Loss(), LCL(), .01, True, SSIMLoss(10))
        average, ema_count, _ = evaluate_pairs(CenterIdentity(), loader, '2.5d', 'fp32', 'cpu',
            nn.L1Loss(), LCL(), .01, True, SSIMLoss(10), reference=False)
        self.assertAlmostEqual(raw['l1'], 4., places=5)
        self.assertAlmostEqual(raw['l2'], 4., places=5)
        self.assertAlmostEqual(raw['total'], 2+.5*raw['lcl'], places=6)
        self.assertEqual(average['total'], raw['total'])
        self.assertEqual(count, ema_count)
        self.assertIsNotNone(preview)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         'Gloo is unavailable')
    def test_three_cpu_ranks_evaluate_two_samples_without_hanging(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.multiprocessing.spawn(distributed_evaluation_worker, args=(3, temporary),
                                        nprocs=3, join=True)
            results = [json.loads((Path(temporary)/f'rank{rank}.json').read_text()) for rank in range(3)]
        self.assertTrue(results[2]['empty'])
        for result in results:
            self.assertEqual(result['count'], 2)
            self.assertEqual(result['ema_count'], 2)
            self.assertAlmostEqual(result['values']['l1'], 4., places=5)
            self.assertAlmostEqual(result['ema']['total'], result['values']['total'], places=6)
            self.assertEqual(result['values'], results[0]['values'])


if __name__ == '__main__':
    unittest.main()
