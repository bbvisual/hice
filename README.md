# Detecting Precise Hand Touch Moments in Egocentric Video

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![CVPR 2026](https://img.shields.io/badge/CVPR%202026-Findings-blue)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.12343)

PyTorch implementation of HiCE for touch event spotting in egocentric video. Given an egocentric video, HiCE detects the precise frame of initial hand-object contact within a tolerance of 2 frames.


## Environment

```bash
conda create -n hice python=3.11
conda activate hice
```

Then install dependencies:

```bash
bash install_env.sh
```

Or manually:

```bash
pip install numpy==1.26.4 pandas==2.3.3
pip install opencv-python-headless==4.9.0.80
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.3 pydantic wandb matplotlib tabulate
```


## Data Preparation

### TouchMoment

Download the TouchMoment dataset (51 GB) from [Box](https://adelaideuniversity.box.com/s/q633lrm5v5trrrj4ddvdwfcpmfupvitj). The dataset is structured as:

```
TouchMoment/
  Frames/
    video_001/
      000000.jpg
      000001.jpg
      ...
    video_002/
      ...
  hand_anno.json
```

Label splits (`train.json`, `val.json`, `test.json`) and the class file are already provided under `data/TouchMoment/`.

### Your Own Dataset

**Frames** — extract video frames into one sub-folder per video. Frame filenames must be zero-padded 6-digit `.jpg` files:

```
Frames/
  video_001/
    000000.jpg
    000001.jpg
    ...
```

**Hand annotations** — HiCE requires a `hand_anno.json` file mapping each frame to left/right hand bounding boxes. Clone and set up [Hands23](https://github.com/ddshan/hand_object_detector), then copy `hands23_script/run_hands23.py` into the Hands23 directory and run:

```bash
python run_hands23.py \
    --frames /path/to/Frames \
    --output     /path/to/hand_anno.json
```

The resulting file maps each video and frame to hand bounding boxes and grasp-type scores:

```json
{
  "video_001": {
    "000000.jpg": {
      "left hand":  {"box": [x1, y1, x2, y2], "grasp": {"NP-Palm": 0.0, "NP-Fin": 1.0, "Pow-Pris": 0.0, ...}},
      "right hand": {"box": [x1, y1, x2, y2], "grasp": {"NP-Palm": 0.0, "NP-Fin": 1.0, "Pow-Pris": 0.0, ...}}
    }
  }
}
```

- `box` is in `[x1, y1, x2, y2]` format. Use `null` when a hand is not visible.
- `grasp` scores (8 types) are required. Only `inference.py` can run without them.

**Label files** — provide `train.json`, `val.json`, and `test.json` under `data/<dataset>/`. Each is a JSON array:

```json
[
  {
    "video": "video_001",
    "num_frames": 300,
    "fps": 15,
    "events": [
      {"frame": 57,  "label": "touch", "comment": ""},
      {"frame": 178, "label": "touch", "comment": ""}
    ]
  }
]
```

`events` may be an empty list `[]` for inference without ground truth.

**Class file** — `data/<dataset>/class.txt`, one class name per line:

```
touch
```


## Pretrained Weights

We provide checkpoints trained on TouchMoment and its two subsets (HOI4D, TACO):

| Dataset | Checkpoint |
|---|---|
| TouchMoment | [touchmoment_checkpoint.pt](https://adelaideuniversity.box.com/s/zv51wrjxuhtbz5sc5smy1kysrvct43sk) |
| HOI4D | [hoi4d_checkpoint.pt](https://adelaideuniversity.box.com/s/c36hidtwbnoqdwcyfw7xqfigz6bgsmq3) |
| TACO | [taco_checkpoint.pt](https://adelaideuniversity.box.com/s/9k1upy08otonop73z2sdlv9rqsrfpbq7) |

Download and use with `--checkpoint` in the [Inference](#inference) section.


## Training

### 1. Configure

Experiment configs live in `config/<dataset>/<dataset>_*.yaml`. See `config/HOI4D/HOI4D_base.yaml` for a complete example.

> **Note:** the dataset name must not contain underscores — e.g. `HOI4D-small_base` is valid but `HOI4D_small_base` is not.

### 2. Store clips (first run only)

Pre-compute and cache clip partitions to `store_dir`. This only needs to be done once:

```bash
python main.py --model <config_name> --train --store
```

### 3. Train

```bash
python main.py --model <config_name> --train
```

Checkpoints are saved to `save_dir/<config_name>/checkpoint_best.pt`.

Optional flags:

| Flag | Description |
|---|---|
| `--resume` | Resume from `checkpoint_last.pt` |
| `--compile` | Use `torch.compile` for faster training |
| `--wandb` | Enable Weights & Biases logging |
| `--gpu` | GPU IDs to use |

### 4. Test

```bash
python main.py --model <config_name> --test
```


## Inference

Run a trained checkpoint on new videos without a full training setup:

```bash
python inference.py \
    --checkpoint checkpoints/touchmoment_checkpoint.pt \
    --frame      demo/demo_video \
    --hand_anno  demo/hand_anno.json \
    --out        demo/output/
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--label` | auto-detect | Label JSON with video names and `num_frames`. Omit to auto-detect videos from `--frame`. Provide with ground-truth events to compute mAP. |
| `--config` | `config/demo/demo.yaml` | Model config YAML. |
| `--threshold` | `0.5` | Score threshold for `pred_touch.json`. |
| `--gpu` | env | GPU id. Respects `CUDA_VISIBLE_DEVICES` if omitted. |

### Outputs

```
output/
  pred_touch.json          # Touch events with score >= threshold (keys: no_nms / nms / snms)
  mAP_calculation/
    pred.json              # Raw high-recall predictions (score >= 0.01)
    pred_nms.json          # After NMS
    pred_snms.json         # After soft-NMS
  results.txt              # mAP table (only when ground-truth events are provided)
```

`pred_touch.json` format:

```json
{
  "threshold": 0.5,
  "no_nms": [{"video": "video_001", "events": [{"label": "touch", "frame": 57, "score": 0.82}]}],
  "nms":    [...],
  "snms":   [...]
}
```

### Re-evaluating mAP from saved predictions

```bash
python eval_from_json.py mAP_calculation/pred.json data/<dataset>/test.json
```


## Config Reference

| Parameter | Description |
|---|---|
| `frame_dir` | Path to extracted frames directory |
| `save_dir` | Directory for checkpoints and results |
| `store_dir` | Directory for cached clip partitions |
| `dataset` | Dataset name (must match `data/<dataset>/`) |
| `clip_len` | Number of frames per clip |
| `crop_dim` | Spatial crop size (224) |
| `feature_arch` | CNN backbone (`rny008_gsf`, `rny002_gsf`) |
| `temporal_arch` | Temporal module (`ed_sgp_mixer`, `mstcn`, `asformer`, `gru`) |
| `n_layers` | Number of temporal layers |
| `radi_displacement` | Displacement radius for sub-frame precision |
| `grasp_loss` | Enable grasp auxiliary loss (requires grasp annotations) |
| `soft_labels` | Enable Gaussian soft labels |
| `bi_interp_post` | Use bilinear interpolation in post-processing |
| `temporal_shift` | Enable GSF temporal shift modules |
| `criterion` | Validation criterion (`map` or `loss`) |
| `num_epochs` | Number of training epochs |
| `learning_rate` | Initial learning rate |
| `loss.type` | Loss function (`ce` or `focal`) |


## Citation

If you use HiCE in your research, please cite:

```bibtex
@InProceedings{han_hice_cvpr_2026,
  author = {Huy Anh Nguyen and Feras Dayoub and Minh Hoai},
  title = {Detecting Precise Hand Touch Moments in Egocentric Video},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year = {2026},
}
```


## Acknowledgements

This codebase builds on [T-DEED](https://github.com/arturxe2/T-DEED) (Temporal Detection of Every Event Displacement). Hand detection and grasp classification use [Hands23](https://github.com/ddshan/hand_object_detector).
