import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
import numpy as np
import torch.distributed as dist
from .layers import conv_layer, deconv_layer
from .adapter import ReciprocalVisionLanguageAdapter
import os
from functools import partial



class Fusion(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                 fusion_adapter="legacy",
                 adapter_layers=(),
                 adapter_visual_dim=768,
                 adapter_text_dim=512,
                 adapter_hidden_dim=128,
                 adapter_heads=8,
                 adapter_pool_size=8,
                 adapter_dropout=0.0,
                 input_is_clip_normalized=True,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        self.initialize_parameters()

        self.adapter_type = str(fusion_adapter).strip().lower()
        if self.adapter_type not in {"legacy", "reciprocal"}:
            raise ValueError(f"Unknown fusion_adapter: {fusion_adapter!r}")
        self.adapter_layers = tuple(sorted(set(int(v) for v in adapter_layers)))
        if self.adapter_type == "reciprocal" and not self.adapter_layers:
            raise ValueError("reciprocal fusion requires at least one adapter layer")
        if self.adapter_layers and (
            self.adapter_layers[0] < 0
            or self.adapter_layers[-1] >= min(self.num_layers, self.dino_layers)
        ):
            raise ValueError("reciprocal adapter layer is outside the paired backbones")
        self.reciprocal_adapters = nn.ModuleDict(
            {
                str(layer): ReciprocalVisionLanguageAdapter(
                    visual_dim=int(adapter_visual_dim),
                    text_dim=int(adapter_text_dim),
                    hidden_dim=int(adapter_hidden_dim),
                    num_heads=int(adapter_heads),
                    pool_size=int(adapter_pool_size),
                    dropout=float(adapter_dropout),
                )
                for layer in self.adapter_layers
            }
            if self.adapter_type == "reciprocal"
            else {}
        )
        self.input_is_clip_normalized = bool(input_is_clip_normalized)
        self.register_buffer(
            "clip_mean",
            torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def _dino_input(self, image):
        image = image.float()
        if self.input_is_clip_normalized:
            image = image * self.clip_std + self.clip_mean
        return (image - self.imagenet_mean) / self.imagenet_std

    def _forward_reciprocal(self, img, text, txt_backbone, dino):
        padding_mask = text.eq(0)
        txt = txt_backbone.token_embedding(text).type(txt_backbone.dtype)
        txt = txt + txt_backbone.positional_embedding.type(
            txt_backbone.dtype
        )[: txt.size(1)]
        txt = txt.permute(1, 0, 2)
        txt_blocks = txt_backbone.transformer.resblocks

        net_input = self._dino_input(img)
        batch, _, image_h, image_w = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat(
            (dino.cls_token.expand(batch, -1, -1), dino_f), dim=1
        )
        dino_f = dino_f + dino.interpolate_pos_encoding(
            dino_f, image_h, image_w
        )
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(batch, -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        patch_size = int(dino.patch_size)
        visual_grid = (image_h // patch_size, image_w // patch_size)
        features_dino = []

        for index in range(max(self.num_layers, self.dino_layers)):
            if index < self.num_layers:
                txt = txt_blocks[index](txt)
            if index < self.dino_layers:
                dino_f = dino.blocks[index](dino_f)
            if index in self.adapter_layers:
                text_tokens = txt.permute(1, 0, 2)
                dino_f, text_tokens = self.reciprocal_adapters[str(index)](
                    dino_f,
                    text_tokens,
                    padding_mask,
                    visual_grid=visual_grid,
                    special_tokens=1 + int(dino.num_register_tokens),
                )
                txt = text_tokens.permute(1, 0, 2)
            if index < self.dino_layers and index in self.output_dinov2:
                features_dino.append(dino_f)

        txt = txt_backbone.ln_final(
            txt.permute(1, 0, 2)
        ).type(txt_backbone.dtype)
        state = txt[
            torch.arange(txt.shape[0], device=txt.device),
            text.argmax(dim=-1),
        ] @ txt_backbone.text_projection

        features_dino.append(dino.norm(dino_f))
        vis_outs = []
        for feature_dino in features_dino:
            patches = feature_dino[:, 1 + int(dino.num_register_tokens):]
            if patches.shape[1] != visual_grid[0] * visual_grid[1]:
                raise RuntimeError(
                    "DINO output token count does not match the reciprocal grid"
                )
            feature_map = patches.reshape(
                batch, visual_grid[0], visual_grid[1], patches.shape[-1]
            ).permute(0, 3, 1, 2)
            vis_outs.append(feature_map)
        return vis_outs, txt, state

    def forward(self, img, text, txt_backbone,dino):
        if self.adapter_type == "reciprocal":
            return self._forward_reciprocal(img, text, txt_backbone, dino)

        B=img.shape[0]
        img = img.type(txt_backbone.dtype)
        vis_outs = []
        outputs=[]
        txt = txt_backbone.token_embedding(text).type(
            txt_backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = txt_backbone.transformer
        txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # BLD -> LBD
        
        #dinov2  
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(dino_f.shape[0], -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        features_dino=[]
        for i in range(self.num_layers):
            txt = txt_enc.resblocks[i](txt)

        # language
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_backbone.text_projection# get sentence-level feature Fs
        
        for i in range(self.dino_layers):
            dino_f = dino.blocks[i](dino_f, txt)
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)
        
        for i, feature_dino in enumerate(features_dino):
            feature_dino=feature_dino[:, 4 + 1 :]
            B,L,C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)

            vis_outs.append(feature_dino)
 

        # forward

        output = vis_outs , txt, state

        return output



