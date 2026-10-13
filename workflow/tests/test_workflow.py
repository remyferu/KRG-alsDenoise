"""Workflow contracts with tiny files and mocks; no reconstruction or training."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workflow/src'))
import h5py
import numpy as np
import tifffile
import yaml
from krg_workflow import api, cleanup, cor, queue, reconstruction, runtime
from krg_workflow.files import json_write, exclusive_lock, safe_child
from krg_workflow.plans import freeze_run, read_plan, reconstruction_identity, stage_signature
from krg_workflow.presets import training_settings
from krg_workflow.scans import projection_view


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'/'scan.h5'
        self.raw.parent.mkdir()
        self.data = np.arange(8*5*32,dtype=np.float32).reshape(8,5,32)
        with h5py.File(self.raw,'w') as f:
            f['exchange/data'] = self.data
            f['exchange/data_white'] = np.ones((1,5,32),dtype=np.float32)
            f['exchange/data_dark'] = np.zeros((1,5,32),dtype=np.float32)
            f['exchange/theta'] = np.linspace(0,180,8)
            for key,value in {
                'measurement/instrument/detector/dimension_x':32,
                'measurement/instrument/detector/dimension_y':5,
                'measurement/instrument/detector/pixel_size':.001,
                'measurement/instrument/monochromator/energy':20000,
                'process/acquisition/rotation/num_angles':8,
                'process/acquisition/rotation/range':180,
            }.items():
                f[key] = [value]
        backend = self.root/'microct'/'backend'
        backend.mkdir(parents=True)
        for name in ('ALS_recon_functions.py','ALS_recon_helper.py'):
            (backend/name).write_text('# fixture\n')
        self.workspace = api.Workspace.create(backend.parent,['/n2i/bin/denoise'],self.root/'scratch')
        with contextlib.redirect_stdout(io.StringIO()):
            self.run = api.inspect_run(self.workspace,self.raw,'test')
        record = dict(name='als_modern',metadata_adapter='optional_camera_distance_v1',directory=str(backend),
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in backend.iterdir()})
        self.run.configure(record,dict(method='fbp',fc=.75,als_chunk_sino=2,use_gpu=False,
                                     preprocessing={},postprocessing={}))
        captured = self.run.snapshot()
        self.plan = freeze_run(captured,self.run.info,0.,dict(center=0.,method='numeric_override',context=captured),context=self.run)

    def write_stacks(self):
        for folder in api.reconstruction_paths(self.run):
            folder.mkdir(exist_ok=True)
            for row in range(5):
                tifffile.imwrite(folder/f'recon_{row:05d}.tiff',np.full((32,32),row,dtype=np.float32))

    def write_config(self):
        full,even,odd = api.reconstruction_paths(self.run)
        cfg = dict(dataset=dict(directory_to_reconstructions=str(self.run.run_dir),
            full_recon_name=full.name,sub_recon_name0=even.name,sub_recon_name1=odd.name),
            train=training_settings(psz=16),infer=dict(window='cosine',overlap=.5))
        path = self.run.run_dir/(full.name+'_v3_5_1_config.yaml')
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_legacy_plans_restore_without_rewriting_identity(self):
        expected = reconstruction_identity(self.plan)
        for version in ('3.0','3.1','3.2','3.3','3.4','3.5'):
            saved = copy.deepcopy(self.plan)
            saved['provenance']['notebook_version'] = version
            json_write(self.run.plan_path,saved)
            restored, result = cor.automatic_cor_workflow(self.run.snapshot(),self.run.info,context=self.run)
            self.assertIsNone(result)
            self.assertEqual(restored,saved)
            self.assertEqual(reconstruction_identity(restored),expected)
            self.assertEqual(stage_signature(restored,'even'),stage_signature(self.plan,'even'))

    def test_context_rejects_changes_and_previous_selection(self):
        captured = self.run.snapshot()
        settings = dict(self.run.settings,fc=.5)
        self.run.configure(self.run.backend,settings)
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.run.assert_current(captured)
        self.run.invalidate()
        with self.assertRaisesRegex(ValueError,'replaced'):
            self.run.snapshot()

    def test_virtual_views_retain_independent_projections(self):
        for parity in (0,1):
            path = projection_view(self.raw,self.run.run_dir/f'view{parity}.h5',parity,self.run.source)
            with h5py.File(path) as f:
                np.testing.assert_array_equal(f['exchange/data'][:],self.data[parity::2])
                self.assertTrue(f['exchange/data'].is_virtual)

    def test_vo_uses_three_rows_native_center_and_midpoint_offset(self):
        centers = iter((16.2,16.4,16.3))
        def native(sino,smin,smax,srad,step,ratio,drop,ncore):
            self.assertEqual(sino.shape,(8,32))
            self.assertEqual(ncore,1)
            return next(centers)
        def read(source,proj,sino,**kwargs):
            return self.data[:,sino,:],-np.deg2rad(np.linspace(0,180,8))
        helper = SimpleNamespace(als=SimpleNamespace(tomopy=SimpleNamespace(find_center_vo=native),read_data=read))
        with patch.object(cor,'load_als',return_value=helper),patch.object(cor,'require_compute_session'):
            result = cor.find_vo_center(self.run.snapshot(),context=self.run)
        self.assertEqual(result['status'],'auto_accepted')
        self.assertAlmostEqual(cor.checked_auto_center(result,self.run.snapshot(),context=self.run),.3)

    def test_reconstruction_stages_resume_with_same_signatures(self):
        stages = []
        def fake_stage(source,output,settings,center,rows,backend,log):
            stages.append(Path(output).name)
            Path(output).mkdir(exist_ok=True)
            for row in range(rows):
                tifffile.imwrite(Path(output)/f'recon_{row:05d}.tiff',np.zeros((32,32),dtype=np.float32))
        with patch.object(reconstruction,'load_als'),patch.object(reconstruction,'run_als_stage',side_effect=fake_stage):
            self.assertEqual(reconstruction.run_plan(self.plan),'completed')
            self.assertEqual(reconstruction.run_plan(self.plan),'already-complete')
        self.assertEqual(len(stages),3)

    def test_queue_repair_and_interruption_preserve_entry_ids(self):
        path = self.workspace.queue_path
        original = queue.enqueue_plan(path,self.run.plan_path)
        self.assertEqual(queue.enqueue_plan(path,self.run.plan_path),original)
        with self.assertRaises(KeyboardInterrupt):
            queue.execute_queue(path,runner=lambda plan: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertFalse(queue.queue_lock(path).exists())
        report = json.loads((path.parent/'queue_report_v3.json').read_text())
        self.assertTrue(report['interrupted'])
        self.assertEqual(report['outcomes'][0]['id'],original['entries'][0]['id'])
        proposal = queue.propose_queue_edit(path,remove_ids=[original['entries'][0]['id']])
        changed = copy.deepcopy(original)
        changed['entries'].append(dict(id='different',plan_path=str(self.run.plan_path)))
        json_write(path,changed)
        with self.assertRaisesRegex(ValueError,'changed after inspection'):
            queue.apply_queue_edit(path,proposal)
        proposal = queue.propose_queue_edit(path,remove_ids=['different'])
        backup = queue.apply_queue_edit(path,proposal)
        self.assertTrue(backup.is_file())
        self.assertEqual(queue.load_queue(path),original)

    def test_cleanup_checks_confirmation_manifest_and_protected_paths(self):
        folder = self.run.run_dir/'auto_cor_vo'
        folder.mkdir()
        (folder/'result.json').write_text('{}')
        protected = self.workspace.protected_paths(self.run)
        preview = cleanup.cleanup_manifest(self.plan,self.workspace.output_root,['cor'],protected_paths=protected)
        self.assertEqual(preview['file_count'],1)
        result = cleanup.execute_cleanup(self.plan,self.workspace.output_root,preview,protected_paths=protected)
        self.assertEqual(result['status'],'dry-run')
        with self.assertRaisesRegex(ValueError,'explicit opt-in'):
            cleanup.execute_cleanup(self.plan,self.workspace.output_root,preview,False,'',False)
        with self.assertRaisesRegex(ValueError,'raw inputs'):
            cleanup.cleanup_manifest(self.plan,self.workspace.output_root,['cor'],protected_paths=[folder])
        (folder/'result.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'changed since preview'):
            cleanup.execute_cleanup(self.plan,self.workspace.output_root,preview,False,str(self.run.run_dir),True)
        preview = cleanup.cleanup_manifest(self.plan,self.workspace.output_root,['cor'])
        result = cleanup.execute_cleanup(self.plan,self.workspace.output_root,preview,False,str(self.run.run_dir),True)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(folder.exists())
        self.assertTrue(self.raw.exists())
        with self.assertRaises(ValueError):
            safe_child(self.run.run_dir,'../raw')

    def test_config_resume_commands_and_disabled_training(self):
        self.write_stacks()
        config = self.write_config()
        def fake_logged(parts,log,cwd,**kwargs):
            self.assertIn('-h',parts)  # Only help, never a training command.
            Path(log).parent.mkdir(exist_ok=True)
            with Path(log).open('a') as stream:
                stream.write('--mode --resume --no-search --architecture')
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),patch.object(api,'run_logged',side_effect=fake_logged):
            training = api.configure_training(self.run,self.workspace,overrides={'psz':16,'validation':{'enabled':False}})
            command = api.training_command(self.run,self.workspace)
            self.assertNotIn('--resume',command)
            training.output.mkdir()
            (training.output/'resume.pth').write_bytes(b'fixture-not-model')
            saved = yaml.safe_load(config.read_text())
            saved['dataset'].update(mean4norm=1.,std4norm=2.)
            saved['train']['patience'] = 71
            config.write_text(yaml.safe_dump(saved))
            before = config.read_bytes()
            api.configure_training(self.run,self.workspace,overrides={'psz':16,'patience':100})
            self.assertEqual(config.read_bytes(),before)
            self.assertIn('--resume',api.training_command(self.run,self.workspace))
            (training.output/'best_val_model.pth').write_bytes(b'fixture-not-model')
            infer = api.inference_command(self.run,self.workspace,'slice',checkpoint='val',row=2)
            self.assertEqual(infer[infer.index('--gpus')+1],'0')
            self.assertEqual(infer[-2:],['--slice-number','2'])
            self.assertTrue(str(api.denoised_path(self.run,'volume')).endswith('_denoised_volume_2.5d_nafnet25d_v3_5_1_ema'))
            self.assertIsNone(api.train_run(self.run,self.workspace,enabled=False))
        with patch.object(api,'run_logged') as launcher:
            api.train_run(None,None,enabled=False)
            launcher.assert_not_called()

    def test_fresh_training_labels_preserve_previous_configs_and_outputs(self):
        self.write_stacks()
        full, even, odd = api.reconstruction_paths(self.run)
        old_config = self.run.run_dir/(full.name+'_config.yaml')
        old_config.write_text('previous training YAML')
        old_output = self.run.run_dir/'TrainOutput'
        old_output.mkdir()
        (old_output/'best_val_model.pth').write_bytes(b'previous model')
        def fake_logged(parts, log, cwd, **kwargs):
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            if '-h' in parts:
                with Path(log).open('a') as stream:
                    stream.write('--mode --resume --no-search --architecture')
            else:
                self.assertIn('prepare', parts)
                config = Path(parts[parts.index('--config-output')+1])
                config.write_text(yaml.safe_dump(dict(dataset=dict(
                    directory_to_reconstructions=str(self.run.run_dir), full_recon_name=full.name,
                    sub_recon_name0=even.name, sub_recon_name1=odd.name), train={}, infer={})))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1'), patch.object(api, 'run_logged', side_effect=fake_logged):
            first = api.configure_training(self.run, self.workspace, label='first',
                overrides={'psz':16, 'validation':{'enabled':False}})
            second = api.configure_training(self.run, self.workspace, label='second',
                overrides={'psz':16, 'validation':{'enabled':False}})
            command = api.training_command(self.run, self.workspace)
        self.assertNotEqual(first.config, second.config)
        self.assertNotEqual(first.log_path, second.log_path)
        self.assertEqual(command[command.index('--output-dir')+1], str(second.output))
        self.assertEqual(old_config.read_text(), 'previous training YAML')
        self.assertEqual((old_output/'best_val_model.pth').read_bytes(), b'previous model')

    def test_runtime_subprocess_agg_and_clean_live_tail(self):
        lines = []
        log = self.root/'command.log'
        command = [sys.executable,'-c',"import os; print(os.environ['MPLBACKEND']); print('\\x1b[92mhello\\x1b[0m')"]
        runtime.run_logged(command,log,self.root,on_line=lines.append)
        self.assertIn('Agg',log.read_text())
        self.assertIn('\x1b[92m',log.read_text())
        self.assertTrue(all('\x1b' not in text for text in lines))

    def test_runtime_interrupt_stops_child_and_failing_command_raises(self):
        process = SimpleNamespace(stdout=io.StringIO('line\n'),pid=98765,
                                  poll=lambda:None,terminate=lambda:None,wait=lambda **kw:0,kill=lambda:None)
        with patch.object(runtime.subprocess,'Popen',return_value=process),patch.object(runtime.os,'killpg',create=True) as stop:
            with self.assertRaises(KeyboardInterrupt):
                runtime.run_logged(['fixture'],self.root/'interrupt.log',self.root,
                    on_line=lambda text: (_ for _ in ()).throw(KeyboardInterrupt()) if text!='Starting…' else None)
            if os.name=='posix':
                stop.assert_called_once()
        with self.assertRaises(subprocess.CalledProcessError):
            runtime.run_logged([sys.executable,'-c','raise SystemExit(3)'],self.root/'failed.log',self.root,on_line=lambda text:None)

    def test_cpu_limit_respects_allocation(self):
        with patch.dict(os.environ,SLURM_CPUS_PER_TASK='16'),patch.object(runtime.os,'cpu_count',return_value=128):
            self.assertLessEqual(runtime.available_cores_minus(4),12)
        with patch.dict(os.environ,SLURM_CPUS_PER_TASK='2'):
            self.assertEqual(runtime.available_cores_minus(4),1)

    def test_zoom_crops_keep_native_pixels_and_context_checks(self):
        from PIL import Image
        from krg_workflow.notebook_ui import image_zoom_bounds, image_view_png, zoom_image_viewer
        image = np.arange(32*32,dtype=np.float32).reshape(32,32)
        bounds = image_zoom_bounds(image.shape,zoom=4,center=(16,16))
        self.assertEqual((bounds[1]-bounds[0],bounds[3]-bounds[2]),(8,8))
        png = image_view_png(image,bounds,(0,1023))
        with Image.open(io.BytesIO(png)) as rendered:
            self.assertEqual(rendered.size,(8,8))
        captured = self.run.snapshot()
        self.run.invalidate()
        with self.assertRaises(ValueError):
            zoom_image_viewer([image],captured=captured,context=self.run)

    def test_lightweight_imports_without_torch_or_notebook_dependencies(self):
        command = [sys.executable,'-c',"import krg_workflow, krg_workflow.api, denoise; import sys; assert 'torch' not in sys.modules; assert 'IPython' not in sys.modules"]
        subprocess.run(command,cwd=ROOT,check=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'workflow/src')))


if __name__=='__main__':
    unittest.main()
