# Face2Parameter Training Tasks

- [ ] Confirm parameter dimensionality by opening your target `param.json` and counting the continuous parameter fields to decide whether the pipelines should use 159 or 223 dimensions (≈10 min).
- [ ] Update `config.py` and the scripts that re-declare dataset paths (`train_myimitator.py`, etc.) so they point to the accessible dataset directories on this machine (≈10 min).
- [ ] Create or verify the required local folders (`./logs/`, `./output/preview/`, `./output/imitator/`, dataset subfolders) so training jobs can write checkpoints and previews without crashing (≈5 min).
- [ ] Extend `requirements.txt` with all runtime dependencies you’ll install (`opencv-python`, `dlib`, `tqdm`, etc.) and run a quick `pip install -r requirements.txt` in the project environment (≈10 min).
- [ ] Gather the missing pretrained assets and drop them into `./checkpoint/` (`LightCNN_29Layers_V2_checkpoint.pth.tar`, `shape_predictor_68_face_landmarks.dat`, `resnet18-5c106cde.pth`, updated imitator weights) or adjust the config to use available files (≈15 min).
- [ ] Download the MobilenetV2 weights referenced in `config.model_urls` and save them locally, then change the config to load from the local path to avoid blocked network calls (≈10 min).
- [ ] Patch `utils.eval_output` (and any other callers) so evaluation no longer depends on the commented-out `config.train_set`—e.g., make writing optional or guard it behind a flag (≈10 min).
- [ ] Prepare the translator dataset by placing aligned face images under `./data/`, confirm the `split_dataset` helper sees them, and run a short dry-run of `train_translator.py` to ensure dataloaders build successfully (≈15 min).
- [ ] Run a small sanity check (1-2 iterations) of `train_myimitator.py` with the updated paths and parameter size to confirm the generator forward/backward pass works without shape or file errors (≈15 min).
