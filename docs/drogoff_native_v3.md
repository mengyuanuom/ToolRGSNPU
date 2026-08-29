# DROG-OFF Native V3

Native V3 is the second-stage DINOv2/CLIP adaptation path. It keeps the public
DROG-OFF dense grasp output contract, but disables DETRIS `DenseAligner` and
`TextAdapter` modules.

The native path provides:

- DINOv2 attention LoRA at layers 2, 5, 8, and 11;
- CLIP text FFN LoRA at the corresponding layers;
- padding-aware, gated visual-to-text cross-attention at every stage;
- separate ImageNet normalization for the DINOv2 image stream;
- a real 64/32/16/8 spatial feature pyramid reassembled from ViT stages;
- direct patch-text alignment supervision from the referred-object mask;
- the existing segmentation, quality, sine, cosine, width, and offset heads.

Train the full RealVLG profile on eight NPUs with:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/realvlg/native_v3_drogoff.yaml
```

The profile uses global batch size 128, 36 epochs, dense Offset V2 targets,
gradient clipping at 1.0, and full GraspNet-VLG training data.

Native V3 checkpoints are structurally different from DROG-OFF V1/V2
checkpoints. Start from the configured DINOv2 and CLIP pretrained weights; do
not resume a V2 optimizer/checkpoint into this profile.
