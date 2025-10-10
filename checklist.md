# Face2Parameter Reproduction Checklist

## Paper Implementation Coverage
- [x] Imitator generator implemented with stacked transposed convolutions (`imitator.py`) consistent with the Face-to-Parameter paper.
- [x] Translator network with residual attention blocks matches the Fast and Robust F2P design (`translator.py`).
- [x] Translator training loop applies LightCNN feature loss, face parsing loss, and parameter cycle loss as described (`train_translator.py`).
- [ ] Training schedules mirror reported settings (current translator script runs for 500 epochs with Adam, paper reports ≈20 epochs; needs confirmation/adjustment).
- [ ] Dataset parameter dimensionality verified (code toggles between 159 and 223 dims; JSON schema not included to confirm correctness).

## Blocking Prerequisites for Training
- [ ] Update `config.py` dataset paths from Windows drive letters (`F:/...`) to accessible locations and supply `param.json`, `face_train/`, and `face_val/`.
- [ ] Populate `./data/` with aligned face photos for translator training (`train_translator.py`).
- [ ] Provide all required checkpoints:
  - `checkpoint/epoch_340_0.434396.pt` (or adjust `config.imitator_model` to an available weight).
  - `checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar`.
  - `checkpoint/shape_predictor_68_face_landmarks.dat`.
  - `checkpoint/resnet18-5c106cde.pth`.
- [ ] Mirror/download the pretrained backbone specified in `config.model_urls` to avoid network fetch (current environment blocks external downloads).
- [ ] Resolve `utils.eval_output` reliance on missing `config.train_set` (currently raises `AttributeError` during evaluation).
- [ ] Reconcile parameter dimensionality expectations between `config.continuous_params_size` and `train_myimitator.py` (uses 223-dim vectors internally).
- [ ] Extend `requirements.txt` to include runtime dependencies (`opencv-python`, `dlib`, `tqdm`, etc.).
- [ ] Ensure output/log directories referenced in `config.py` (`./logs/`, `./output/preview`, `./output/imitator`) exist before training.
