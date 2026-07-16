# Pretrained weights

ToolRGSNPU does not commit large binary weights. Run commands from the repository
root so the checked-in configs resolve their default paths below `pretrain/`.

## One-command download

List the available artifacts without downloading anything:

```bash
python tools/download_pretrained.py
```

Download only the weights needed by a model family:

```bash
# CROG, CROG-OFF, MapleGrasp, GGCNN-CLIP, GRConvNet-CLIP, LGD
python tools/download_pretrained.py clip-rn50

# DROG and DROG-OFF
python tools/download_pretrained.py clip-vit-b16 dinov2-vitb14-reg4

# ETRG-A R50 or R101 (ResNet-18 is optional for offline depth initialization)
python tools/download_pretrained.py clip-rn50 resnet18
python tools/download_pretrained.py clip-rn101 resnet18

# GraspMamba backbone files
python tools/download_pretrained.py clip-rn50 mambavision-t
```

To cache every supported backbone weight:

```bash
python tools/download_pretrained.py --all
```

The downloader writes to `pretrain/`, skips valid existing files, uses a
temporary `.part` file, and validates the published checksum when one is
available. Use `--output-dir /path/to/pretrain` for another destination and
override the matching `TRAIN.*_pretrain` value in YAML.

## Official direct links

| Config key / purpose | Local filename | Used by | Official source |
| --- | --- | --- | --- |
| `clip_pretrain` | `pretrain/RN50.pt` | CROG family, MapleGrasp, GGCNN-CLIP, GRConvNet-CLIP, LGD, GraspMamba, ETRG-R50 | [OpenAI CLIP RN50](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) |
| `clip_pretrain` | `pretrain/RN101.pt` | ETRG-R101 | [OpenAI CLIP RN101](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt) |
| `clip_pretrain` | `pretrain/ViT-B-16.pt` | DROG and DROG-OFF | [OpenAI CLIP ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) |
| `dino_pretrain` | `pretrain/dinov2_vitb14_reg4_pretrain.pth` | DROG and DROG-OFF | [Meta DINOv2 ViT-B/14 with registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) |
| `mamba_pretrain` | `pretrain/mambavision_tiny_1k.pth.tar` | GraspMamba | [NVIDIA MambaVision-T](https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar) |
| `depth_pretrain` (optional) | `pretrain/resnet18-f37072fd.pth` | ETRG depth encoder | [TorchVision ResNet-18](https://download.pytorch.org/models/resnet18-f37072fd.pth) |

These are backbone initialization weights, not trained ToolRGS grasp-model
checkpoints. Their upstream licenses still apply; in particular, check the
MambaVision code and weight licenses before redistribution or commercial use.

## Training checkpoints

`TRAIN.weight` initializes model parameters from a ToolRGS task checkpoint.
`TRAIN.resume` restores the model, optimizer, scheduler, epoch, and best metrics
from an interrupted run. Neither is required for a new training run, and this
repository currently does not publish universal task checkpoints because they
depend on the selected dataset, split, prompt protocol, and model configuration.

New checkpoints are written under the configured `TRAIN.output_dir`, for
example:

```text
exp/vcot/drogoff_vcot/best_iou_model.pth
exp/vcot/drogoff_vcot/best_jindex_model.pth
exp/vcot/drogoff_vcot/epoch_5.pth
```

Before training, verify all paths without allocating the full model:

```bash
python tools/check_npu_env.py --config config/vcot/drogoff.yaml
```
