# KRG CT workflow helpers

Install this lightweight Python 3.8-compatible package into the working ALS
environment, separate from the PyTorch denoising backend:

```bash
python -m pip install --no-deps -e workflow
krg-workflow doctor
```

See [first-time setup](../docs/SETUP_NERSC.md) and the
[workflow module/API guide](../docs/WORKFLOW.md). The package contains no model
training on import and no Torch dependency. Its CLI can inspect queues, plot
existing history and explicitly execute frozen reconstruction queues.
