"""Include the standalone ALS workflow contracts in the repository suite."""
import importlib.util
from pathlib import Path


def load_tests(loader, tests, pattern):
    path = Path(__file__).resolve().parents[1]/'workflow/tests/test_workflow.py'
    spec = importlib.util.spec_from_file_location('workflow_contract_tests',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return loader.loadTestsFromModule(module)
