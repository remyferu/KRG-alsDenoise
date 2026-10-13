# KRG CT workflow helpers

Python 3.8 package for ALS scan inspection, centering, reconstruction queues,
notebook displays and training-history plots. PyTorch runs in the separate
denoising environment and is not a dependency of this package.

Install from the repository root in an ALS environment with its scientific
dependencies already installed:

```bash
python -m pip install --no-deps -e workflow
krg-workflow doctor
```

The CLI provides environment diagnostics, history plotting, queue inspection
and reconstruction queue execution. See [NERSC setup](../docs/SETUP_NERSC.md)
and the [workflow guide](../docs/WORKFLOW.md).
