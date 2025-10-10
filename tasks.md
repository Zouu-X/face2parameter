# Face2Parameter Training Tasks

- [ ] Confirm the continuous parameter dimensionality by sampling several entries in the new `param.json` and counting elements (log the chosen value for later configs, ≈10 min).
- [ ] Validate that every param key matches an image filename in `images/` (e.g., append `.png` to the key and flag any missing files) so dataloaders won’t break mid-epoch (≈15 min).
- [ ] Produce reproducible train/val splits from the 70 512 images—write a helper (or script) that shuffles keys and writes two index files, instead of physically copying images (≈15 min).
- [ ] Refactor `dataset.Imitator_Dataset` to read from the new key/index files rather than numeric filenames, and make it accept the split lists created above (≈15 min).
- [ ] Update `config.py`, `train_myimitator.py`, and any other scripts with hard-coded Windows paths so they reference the local `images/` directory and `param.json` (≈10 min).
- [ ] Create or verify runtime folders (`./logs/`, `./output/preview/`, `./output/imitator/`) so checkpoints and previews can be written without errors (≈5 min).
- [ ] Extend `requirements.txt` with missing dependencies you’ll install (`opencv-python`, `dlib`, `tqdm`, etc.) and run `pip install -r requirements.txt` in the environment (≈10 min).
- [ ] Gather the required pretrained assets into `./checkpoint/` (`LightCNN_29Layers_V2_checkpoint.pth.tar`, `shape_predictor_68_face_landmarks.dat`, `resnet18-5c106cde.pth`, latest imitator weights) or update configs to point at available weights (≈15 min).
- [ ] Mirror the MobilenetV2 checkpoint referenced by `config.model_urls` to a local file and switch the config to load from that offline path (≈10 min).
- [ ] Patch `utils.eval_output` (and related code) to avoid referencing the deprecated `config.train_set`, preventing evaluation crashes (≈10 min).
- [ ] Point the translator pipeline to the new split lists, confirm `split_dataset` (or a replacement) sees the data correctly, and run a short dry-run of `train_translator.py` (≈15 min).
- [ ] Execute a brief sanity run of `train_myimitator.py` using the updated dataset loader to ensure forward/backward passes work and file lookups succeed (≈15 min).
