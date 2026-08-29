# DROG-OFF Native V4

Native V4 is the decoder-free final variant of the DINOv2/CLIP adaptation
line. It keeps the V3 encoder-side LoRA, layer-wise adapters, and supervised
patch-text alignment, but removes the inherited DETRIS-style multimodal query
decoder.

The V4 dense path is:

1. CLIP text and DINOv2 image encoders with selected LoRA updates.
2. Padding-aware layer-wise text-to-visual adaptation.
3. Four-stage visual feature pyramid.
4. Direct valid-token-to-pixel attention at every pyramid scale.
5. Alignment-logit spatial prior and sentence-level FiLM.
6. Existing text-conditioned dense grasp and offset projector.

The predicted alignment map is used as a soft feature prior and is supervised
by the target mask during training. No target mask is used during inference.

## Training

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/realvlg/native_v4_drogoff.yaml
```

V4 uses the full configured training split for 36 epochs. Its checkpoint is
not structurally compatible with V2 or V3 because the query decoder is absent.
Start V4 from the configured CLIP and DINOv2 pretrained weights.

For a controlled ablation, compare V3 and V4 with the same data split, seed,
input size, optimizer, epoch count, and evaluation protocol.
