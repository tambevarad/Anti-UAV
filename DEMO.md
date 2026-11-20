# Run the Anti-UAV Demo

These steps run the bundled sample video with the detector and tracker in `Codes/`.

## 1. Open a terminal in the repo

```bash
cd /home/varad/utari-ais-computer-vision/Anti-UAV
```

If the repo is in a different folder, use that folder instead.

## 2. Create and activate a virtual environment

Skip the create command if you already have a Python 3.8 environment named `venv` in the repo.

```bash
python3.8 -m venv venv
source venv/bin/activate
```

The prompt should now show `(venv)`. Install the demo packages into this environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "opencv-python<4.12" numpy==1.24.4 Pillow==10.4.0 matplotlib==3.7.5 scipy==1.10.1 numba==0.58.1 PyYAML tqdm
```

These commands install the CPU version of PyTorch. A CUDA GPU is not required, but CPU tracking is slower.

## 3. Run the sample

Run this from the repo root, with the virtual environment active:

```bash
python Codes/demo_detect_track_fixed.py --device cpu
```

The demo reads `Codes/testvideo/testvideo/n16.mp4` and loads `Codes/detect_wrapper/weights/best.pt`. The sample video and model checkpoint are large local assets and are not included in this code branch; provide them at those paths before running. It opens a window showing the video and boxes, and prints detection/tracking boxes and timing in the terminal. Let the video finish, or press `Ctrl+C` to stop it early. A desktop display is needed for the OpenCV window.

## Troubleshooting

- `No module named PIL`, `torch`, or `cv2`: activate `venv` and install the packages from step 2 in that same terminal.
- `Video not found` or `Weights not found`: check that both bundled paths above exist.
- The demo is slow: CPU inference works but can take several minutes for the full video.
- OpenCV cannot open a window: run from a desktop session rather than a headless terminal.

## Next Steps

The sample demo is separate from dataset evaluation. For tracking and evaluation scripts, see the main [README](README.md) and the top-level `track_*.py` and `evaluate_*.py` files.
