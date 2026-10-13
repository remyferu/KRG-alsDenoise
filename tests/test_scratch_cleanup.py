"""Scratch deletion checks with disposable test trees; no scientific workloads."""
import ast
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workflow/src'))
from krg_workflow import scratch_cleanup as cleanup


class ScratchCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'scratch'
        self.root.mkdir()
        self.first = self.root / 'scan_a/test_1'
        self.second = self.root / 'scan_b/test_2'
        for path in (self.first, self.second):
            (path / 'nested').mkdir(parents=True)
            (path / 'nested/model.txt').write_bytes(b'weights')
        (self.root / 'queue_v3.json').write_text('{"entries": []}')

    def test_depth_pattern_and_incomplete_runs_are_discovered(self):
        (self.root / 'scan_a/unfinished').mkdir()
        (self.root / '.hidden/private').mkdir(parents=True)
        found = cleanup.discover_tests(self.root)
        self.assertEqual({t['relative'] for t in found['tests']},
                         {str(p.relative_to(self.root)) for p in (self.first, self.second, self.first.parent/'unfinished')})
        self.assertEqual(len(cleanup.discover_tests(self.root, pattern='test*')['tests']), 2)
        self.assertEqual(len(cleanup.discover_tests(self.first.parent, depth=1)['tests']), 2)
        self.assertEqual(len(cleanup.discover_tests(self.root, depth=3)['tests']), 2)

    def test_preview_is_read_only_and_confirmed_deletion_is_scoped(self):
        selected = [str(self.first.relative_to(self.root))]
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        preview = cleanup.preview_deletion(self.root, selected)
        self.assertEqual(preview['total_bytes'], 7)
        self.assertEqual(preview['targets'][0]['files'], 1)
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        for phrase, stopped in (('', False), ('DELETE 2', True), ('DELETE 1', False)):
            with self.assertRaises(ValueError):
                cleanup.delete_tests(preview, phrase, stopped)
        result = cleanup.delete_tests(preview, 'DELETE 1', True)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(self.first.exists())
        self.assertTrue(self.first.parent.exists())
        self.assertEqual((self.second/'nested/model.txt').read_bytes(), b'weights')
        self.assertTrue((self.root/'queue_v3.json').exists())
        report = json.loads(Path(result['report_path']).read_text())
        self.assertEqual(report['deleted'], [str(self.first.resolve())])

    def test_all_selected_runs_are_checked_before_any_deletion(self):
        preview = cleanup.preview_deletion(self.root, [str(p.relative_to(self.root)) for p in (self.first, self.second)])
        (self.second/'new.log').write_text('new training data')
        with self.assertRaisesRegex(ValueError, 'changed since preview'):
            cleanup.delete_tests(preview, 'DELETE 2', True)
        self.assertEqual((self.first/'nested/model.txt').read_bytes(), b'weights')
        self.assertEqual((self.second/'nested/model.txt').read_bytes(), b'weights')
        self.assertFalse((self.first/'.run.lock').exists())
        self.assertFalse((self.second/'.run.lock').exists())

    def test_active_or_stale_locks_block_preview_and_delete(self):
        selected = [str(self.first.relative_to(self.root))]
        preview = cleanup.preview_deletion(self.root, selected)
        lock = self.first/'.run.lock'
        lock.write_text('active')
        self.assertTrue(next(t for t in cleanup.discover_tests(self.root)['tests'] if t['path'] == str(self.first))['locked'])
        with self.assertRaises(RuntimeError):
            cleanup.preview_deletion(self.root, selected)
        with self.assertRaises(RuntimeError):
            cleanup.delete_tests(preview, 'DELETE 1', True)
        self.assertEqual(lock.read_text(), 'active')
        lock.unlink()
        (self.first/'nested/.run.lock').write_text('nested job')
        with self.assertRaises(RuntimeError):
            cleanup.preview_deletion(self.root, selected)

    def test_file_changes_and_replaced_run_identity_are_rejected(self):
        selected = [str(self.first.relative_to(self.root))]
        preview = cleanup.preview_deletion(self.root, selected)
        (self.first/'nested/model.txt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            cleanup.delete_tests(preview, 'DELETE 1', True)
        original = self.first.with_name('old_test')
        self.first.rename(original)
        (self.first/'nested').mkdir(parents=True)
        (self.first/'nested/model.txt').write_bytes(b'weights')
        with self.assertRaises(ValueError):
            cleanup.delete_tests(preview, 'DELETE 1', True)
        self.assertTrue(original.exists())

    def test_escape_root_home_and_duplicate_targets_are_rejected(self):
        for name in ('../outside', str(self.first.resolve()), '.', 'scan_a/../test_1', '.hidden/private'):
            with self.assertRaises((ValueError, OSError)):
                cleanup.preview_deletion(self.root, [name])
        with self.assertRaises(ValueError):
            cleanup.scratch_root(Path(self.root.anchor))
        with self.assertRaises(ValueError):
            cleanup.scratch_root(Path.home())
        with self.assertRaises(ValueError):
            cleanup.preview_deletion(self.root, [str(self.first.relative_to(self.root))]*2)

    def test_linked_paths_are_rejected(self):
        # Mock link detection on Windows hosts where creating symlinks needs privileges.
        original = cleanup.is_link_like
        for blocked in (self.root, self.first.parent, self.first, self.first/'nested/model.txt'):
            with patch.object(cleanup, 'is_link_like', side_effect=lambda p: Path(p) == blocked or original(p)):
                with self.assertRaises(ValueError):
                    cleanup.preview_deletion(self.root, [str(self.first.relative_to(self.root))])
        with patch.object(cleanup, 'is_link_like', side_effect=lambda p: Path(p) == self.first or original(p)):
            found = cleanup.discover_tests(self.root)
            self.assertFalse(any(t['path'] == str(self.first) for t in found['tests']))

    def test_partial_failure_is_recorded_and_other_selected_runs_finish(self):
        preview = cleanup.preview_deletion(self.root, [str(p.relative_to(self.root)) for p in (self.first, self.second)])
        original = cleanup.shutil.rmtree

        def remove(path):
            if Path(path) == self.first/'nested':
                raise PermissionError('fixture permission error')
            return original(path)

        with patch.object(cleanup.shutil, 'rmtree', side_effect=remove):
            result = cleanup.delete_tests(preview, 'DELETE 2', True)
        self.assertEqual(result['status'], 'partial-failure')
        self.assertEqual(len(result['failed']), 1)
        self.assertTrue(self.first.exists())
        self.assertFalse(self.second.exists())
        self.assertTrue(Path(result['report_path']).is_file())
        self.assertFalse((self.first/'.run.lock').exists())

    @unittest.skipUnless(importlib.util.find_spec('ipywidgets') and importlib.util.find_spec('IPython'), 'Notebook UI dependencies are optional.')
    def test_widget_requires_preview_and_invalidates_changed_inputs(self):
        with patch('IPython.display.display'), contextlib.redirect_stdout(io.StringIO()):
            panel = cleanup.cleanup_browser(self.root)
            root, layout, scan, summary, selected, preview_button, reviewed, stopped, phrase, delete, outcome = panel.children
            self.assertTrue(delete.disabled)
            selected.value = (str(self.first.relative_to(self.root)),)
            preview_button.click()
            self.assertTrue(delete.disabled)
            stopped.value = True
            phrase.value = 'DELETE 1'
            self.assertFalse(delete.disabled)
            selected.value = (str(self.second.relative_to(self.root)),)
            self.assertTrue(delete.disabled)
            self.assertFalse(stopped.value)
            self.assertEqual(phrase.value, '')
            preview_button.click()
            stopped.value = True
            phrase.value = 'DELETE 1'
            self.assertFalse(delete.disabled)
            layout.children[1].value = '*test*'
            self.assertTrue(delete.disabled)
            self.assertEqual(selected.options, ())
            preview_button.click()
            self.assertTrue(delete.disabled)
            for child in panel.children:
                child.close()
            panel.close()

    def test_notebook_cells_are_python38_compatible_and_not_destructive_on_run_all(self):
        notebook = json.loads((ROOT/'notebooks/Scratch_Cleanup.ipynb').read_text(encoding='utf-8'))
        cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        for number, cell in enumerate(cells, 1):
            code = ''.join(cell['source'])
            self.assertTrue(code.startswith('# {}.'.format(number)))
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
            tree = ast.parse(code, feature_version=(3, 8))
            called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            self.assertNotIn('delete_tests', called)


if __name__ == '__main__':
    unittest.main()
