# Face2Parameter Training Tasks

- [x] Produce reproducible train/val splits from the 70 512 images—write a helper (or script) that shuffles keys and writes two index files, instead of physically copying images (≈15 min). *(Completed: `tools/create_data_split.py`, current subset 20 952 train / 5 239 val).*
- [x] Refactor `dataset.Imitator_Dataset` to read from the new key/index files rather than numeric filenames, and make it accept the split lists created above (≈15 min). *(Completed: `train_myimitator.py` now loads JSON indices and resizes to 512².)*
- [x] Update `config.py`, `train_myimitator.py`, and any other scripts with hard-coded Windows paths so they reference the local `images/` directory and `param.json` (≈10 min).
- [x] Create or verify runtime folders (`./logs/`, `./output/preview/`, `./output/imitator/`) so checkpoints and previews can be written without errors (≈5 min). *(Handled programmatically when training starts.)*
- [ ] Extend `requirements.txt` with missing dependencies you’ll install (`opencv-python`, `dlib`, `tqdm`, etc.) and run `pip install -r requirements.txt` in the environment (≈10 min).
- [ ] Gather the required pretrained assets into `./checkpoint/` (`LightCNN_29Layers_V2_checkpoint.pth.tar`, `shape_predictor_68_face_landmarks.dat`, `resnet18-5c106cde.pth`, latest imitator weights) or update configs to point at available weights (≈15 min).
- [ ] Mirror the MobilenetV2 checkpoint referenced by `config.model_urls` to a local file and switch the config to load from that offline path (≈10 min).
- [ ] Patch `utils.eval_output` (and related code) to avoid referencing the deprecated `config.train_set`, preventing evaluation crashes (≈10 min).
- [ ] Point the translator pipeline to the new split lists, confirm `split_dataset` (or a replacement) sees the data correctly, and run a short dry-run of `train_translator.py` (≈15 min).
- [x] Execute a brief sanity run of `train_myimitator.py` using the updated dataset loader to ensure forward/backward passes work and file lookups succeed (≈15 min). *(Completed: training currently running on subset.)*
