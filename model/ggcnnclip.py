import torch
import torch.nn as nn
import torch.nn.functional as F
from .crog_clip import build_model

class TextVisualFusionFiLM(nn.Module):

    def __init__(self, vis_dim: int, text_dim: int, hidden_dim: int = 128):
        """
        Args:
            vis_dim:   Number of channels in the visual feature map (C).
            text_dim:  Dimension of the text embedding (D).
            hidden_dim: Hidden size for the MLP that processes text.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Linear(hidden_dim, vis_dim)
        self.beta = nn.Linear(hidden_dim, vis_dim)

    def forward(self, feat, e_txt):
        """
        Args:
            feat:  Tensor of shape (B, C, H, W) - visual feature map.
            e_txt: Tensor of shape (B, D)       - text embedding.

        Returns:
            feat_fused: Tensor of shape (B, C, H, W).
        """
        B, C, H, W = feat.shape

        # Text → hidden representation
        h = self.mlp(e_txt)        # (B, hidden_dim)

        # Map hidden representation to per-channel gamma and beta
        # Raw CLIP states can have a much larger magnitude than the shallow
        # GG-CNN feature stream. Bound both FiLM branches so one batch cannot
        # explode the activations and poison the optimizer with NaN gradients.
        gamma = torch.tanh(self.gamma(h))      # (B, C)
        beta = torch.tanh(self.beta(h))        # (B, C)

        # Reshape to broadcast over H and W
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # FiLM transformation
        feat_fused = feat * (1.0 + gamma) + beta
        return feat_fused

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class GGCNNWithText(nn.Module):
    """
    GG-CNN with text conditioning (e.g. CLIP text embedding).

    This class extends the original GG-CNN by:
      - Adding a FiLM fusion module that conditions the final feature map on text.
      - Modifying forward(...) to take an extra text embedding e_txt.
      - Adding a compute_loss(..., e_txt) that matches the style of GraspModel.
    """

    def __init__(self, input_channels: int = 4, text_dim: int = 512,
                 dropout: bool = False, prob: float = 0.0,
                 predict_short_side: bool = False):
        """
        Args:
            input_channels: Number of input channels (e.g. 4 for RGB-D).
            text_dim:       Dimension of the text embedding (e.g. CLIP text dim).
            dropout:        Unused here, kept for API compatibility.
            prob:           Unused here, kept for API compatibility.
        """
        super().__init__()
        self.predict_short_side = bool(predict_short_side)

        # Encoder
        self.conv1 = nn.Conv2d(
            input_channels, filter_sizes[0],
            kernel_sizes[0], stride=strides[0], padding=3
        )
        self.conv2 = nn.Conv2d(
            filter_sizes[0], filter_sizes[1],
            kernel_sizes[1], stride=strides[1], padding=2
        )
        self.conv3 = nn.Conv2d(
            filter_sizes[1], filter_sizes[2],
            kernel_sizes[2], stride=strides[2], padding=1
        )

        # Decoder
        self.convt1 = nn.ConvTranspose2d(
            filter_sizes[2], filter_sizes[3],
            kernel_sizes[3], stride=strides[3],
            padding=1, output_padding=1
        )
        self.convt2 = nn.ConvTranspose2d(
            filter_sizes[3], filter_sizes[4],
            kernel_sizes[4], stride=strides[4],
            padding=2, output_padding=1
        )
        self.convt3 = nn.ConvTranspose2d(
            filter_sizes[4], filter_sizes[5],
            kernel_sizes[5], stride=strides[5],
            padding=5, output_padding=1
        )

        # Text-visual FiLM fusion: channel dim = filter_sizes[5] = 32
        self.fusion = TextVisualFusionFiLM(
            vis_dim=filter_sizes[5],
            text_dim=text_dim,
            hidden_dim=128
        )

        # Output heads: quality, cos, sin, width
        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        if self.predict_short_side:
            self.short_side_output = nn.Conv2d(
                filter_sizes[5], 1, kernel_size=2
            )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x_in, e_txt):
        """
        Forward pass.

        Args:
            x_in:  Tensor of shape (B, input_channels, H, W) - input image.
            e_txt: Tensor of shape (B, text_dim)             - text embedding.

        Returns:
            pos_output, cos_output, sin_output, width_output
            each of shape (B, 1, H_out, W_out)
        """
        x = F.relu(self.conv1(x_in))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        # Text-conditioned FiLM fusion
        x = self.fusion(x, e_txt)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        outputs = (pos_output, cos_output, sin_output, width_output)
        if self.predict_short_side:
            outputs = (*outputs, self.short_side_output(x))
        return outputs

    def compute_loss(self, xc, yc, e_txt):
        """
        Compute regression losses for grasp prediction.

        Args:
            xc:    Input image, shape (B, input_channels, H, W).
            yc:    Tuple (y_pos, y_cos, y_sin, y_width),
                   each of shape (B, 1, H_out, W_out).
            e_txt: Text embedding, shape (B, text_dim).

        Returns:
            dict with:
              'loss':   total loss
              'losses': dict of individual losses
              'pred':   dict of predicted maps
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)[:4]

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc, e_txt):
        """
        Inference helper.

        Args:
            xc:    Input image, shape (B, input_channels, H, W).
            e_txt: Text embedding, shape (B, text_dim).

        Returns:
            dict with predicted maps:
              'pos', 'cos', 'sin', 'width'
        """
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)[:4]
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class GGCNN_CLIP(nn.Module):
    """GG-CNN + CLIP with native long- and short-side grasp heads."""

    grasp_size_loss_activation = "sigmoid"

    def __init__(self, cfg):
        super().__init__()
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.predicts_grasp_short_side = bool(
            getattr(cfg, "predict_grasp_short_side", False)
        )
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )

        clip_model = torch.jit.load(
            cfg.clip_pretrain, map_location="cpu"
        ).eval()
        print(
            f"[CROG_GGCNN_CLIP] Load pretrained CLIP: "
            f"{self.use_pretrained_clip}"
        )
        self.backbone = build_model(
            clip_model.state_dict(),
            cfg.word_len,
            self.use_pretrained_clip,
        ).float()
        self.grasp_head = GGCNNWithText(
            input_channels=getattr(cfg, "input_channels", 3),
            text_dim=cfg.word_dim,
            predict_short_side=self.predicts_grasp_short_side,
        )

    @staticmethod
    def _resize_target(target, output_size):
        if target is None:
            return None
        if target.shape[-2:] != output_size:
            target = F.interpolate(
                target, output_size, mode="nearest"
            ).detach()
        return target

    def forward(
        self,
        img,
        word,
        ins_mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
        grasp_short_mask=None,
    ):
        del grasp_off_mask, grasp_off_weight
        _, state = self.backbone.encode_text(word)
        state = F.normalize(state.float(), dim=-1)
        head_outputs = self.grasp_head(img, state)
        pos_pred, cos_pred, sin_pred, wid_pred = head_outputs[:4]
        short_pred = (
            head_outputs[4] if self.predicts_grasp_short_side else None
        )

        predictions = (
            pos_pred,
            pos_pred,
            sin_pred,
            cos_pred,
            wid_pred,
        )
        if self.predicts_grasp_short_side:
            predictions = (*predictions, short_pred)

        output_size = pos_pred.shape[-2:]
        targets = tuple(
            self._resize_target(target, output_size)
            for target in (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )
        )
        if self.predicts_grasp_short_side:
            targets = (
                *targets,
                self._resize_target(grasp_short_mask, output_size),
            )

        if not self.training:
            return tuple(item.detach() for item in predictions), targets
        if any(target is None for target in targets):
            raise ValueError(
                "GGCNN-CLIP training requires all enabled dense target maps"
            )

        quality_loss = F.smooth_l1_loss(pos_pred, targets[1])
        sine_loss = F.smooth_l1_loss(sin_pred, targets[2])
        cosine_loss = F.smooth_l1_loss(cos_pred, targets[3])
        width_loss = F.smooth_l1_loss(
            torch.sigmoid(wid_pred), targets[4]
        )
        short_loss = (
            F.smooth_l1_loss(torch.sigmoid(short_pred), targets[5])
            if self.predicts_grasp_short_side
            else None
        )
        total_loss = (
            quality_loss + sine_loss + cosine_loss + width_loss
        )
        if short_loss is not None:
            total_loss = (
                total_loss + self.short_side_loss_weight * short_loss
            )
        zero = quality_loss.detach() * 0.0
        losses = {
            "m_ins": zero,
            "m_qua": quality_loss.detach(),
            "m_sin": sine_loss.detach(),
            "m_cos": cosine_loss.detach(),
            "m_wid": width_loss.detach(),
        }
        if short_loss is not None:
            losses["m_short"] = short_loss.detach()
        return (
            tuple(item.detach() for item in predictions),
            targets,
            total_loss,
            losses,
        )