# Current State — 2025-10-14

## Completed
- Generated JSON-based data splits with `tools/create_data_split.py`; current working subset contains 20 952 train and 5 239 validation samples.
- Refactored `train_myimitator.py` to consume the split indices, resize inputs to 512×512, and auto-create output directories; imitator training is now running successfully on the subset.
- Added notebook- and script-friendly utilities for inspecting parameter vector lengths.

## In Progress / Pending
- Investigate the 108 203 param keys that lacked matching images during split generation and reconcile filenames or paths so the full dataset can be used.
- Confirm continuous parameter dimensionality from the latest `param.json` and propagate the chosen size across configs.
- Update `config.py` and other scripts to remove hard-coded Windows paths in favor of the new dataset location.
- Populate missing pretrained checkpoints and extend `requirements.txt` before moving on to translator training and evaluation fixes.
