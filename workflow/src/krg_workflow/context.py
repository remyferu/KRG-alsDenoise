"""Explicit run state shared by actions and captured widget callbacks."""
import copy
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunContext:
    source: dict
    run_dir: Path
    info: dict
    backend: dict = None
    settings: dict = None
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    valid: bool = True
    automatic_cor: dict = None
    manual_cor: dict = None
    alignment_center: float = None
    training: object = None

    def invalidate(self):
        """Disable old callbacks when selecting a different scan or run."""
        self.valid = False
        self.training = None

    def configure(self, backend, settings):
        if not self.valid:
            raise ValueError('This run was replaced. Rerun Inspect scan.')
        if self.backend != backend or self.settings != settings:
            self.backend, self.settings = copy.deepcopy(backend), copy.deepcopy(settings)
            self.generation = uuid.uuid4().hex
            self.automatic_cor = self.manual_cor = self.alignment_center = None
            self.training = None
        return self.snapshot()

    def snapshot(self):
        if not self.valid or self.backend is None:
            raise ValueError('Run is missing or replaced. Rerun Inspect scan and Configure reconstruction.')
        return copy.deepcopy(dict(source=self.source, run_dir=str(self.run_dir.resolve()),
                                  dimensions=list(self.info['shape']), backend=self.backend,
                                  settings=self.settings, generation=self.generation))

    def assert_current(self, captured, check_source=False):
        if captured != self.snapshot():
            raise ValueError('Stale controls/results: rerun this action for the selected scan.')
        if check_source:
            from .scans import fingerprint
            if fingerprint(self.source['path']) != self.source:
                raise ValueError('Raw source changed; rerun Inspect scan with a new run label.')

    @property
    def plan_path(self):
        return self.run_dir/'run_plan.json'

    @property
    def shape(self):
        return tuple(self.info['shape'])
