from typing import Tuple, Union, List, Any
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class TextAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        conv_block = BasicConv1d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels + ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels + ch1x1 + ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        B, P, D = x0.shape

        x0 = F.relu(x0, inplace=True)

        xs = x0[:, 1:, :].permute(0, 2, 1)  
        dense_branch1 = self.dense_branch1(xs)
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        outputs = torch.cat(outputs, dim=1).permute(0, 2, 1) 

        clstoken = x0[:, 0:1, :]
        outputs = torch.cat([clstoken, outputs], dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x
        return outputs
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.back_proj = nn.Linear(embed_dim, image_dim)
        
    def forward(self, image_features, text_features, attention_mask=None):


        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        query = image_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        
        attn_output = self.back_proj(attn_output)
        
        return attn_output.permute(1, 0, 2)


class DenseAligner(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
        embed_dim=128,
        num_heads=8,
        text_dim=512,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
        conv_block = BasicConv2d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels+ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels+ch1x1+ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

        self.cross = CrossModalAttention(in_channels, text_dim, embed_dim, num_heads)

        self._initialize_weights()

    def _initialize_weights(self):
   
        for module in self.modules():
            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  
            elif isinstance(module, nn.Linear):
  
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)        

    def forward(self, x: Tensor, text_features,split_token=5) -> List[Tensor]:

        x0 = self.D_fc1(x)
        B,P,D = x0.shape
        W = H = int(math.sqrt(P-1))

        x0 = F.relu(x0, inplace=True)
        
        xs = x0[:,split_token:,:]
        # B, W, H, D / 8, 32, 32, 384 -> B, D, W, H / B, 384, 32, 32
        xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
        # B, 192, 32, 32
        dense_branch1 = self.dense_branch1(xs)
        # B, 96, 32, 32
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        # B, 96, 32, 32
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        # B, 384, 32, 32
        outputs = torch.cat(outputs,dim=1) + xs
        # B, 384, 32, 32 -> B, 384, 1024 -> B, 1024, 384
        outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
        # text fusion
        outputs = self.cross(outputs, text_features)

        clstoken =  x0[:,0:split_token,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs

class ReciprocalVisionLanguageAdapter(nn.Module):
    """Bidirectionally connect DINO patches and intermediate CLIP text tokens.

    The legacy DenseAligner repeatedly injects the same final CLIP text into
    several DINO blocks. This adapter is placed between paired DINO and CLIP
    transformer layers. Text attends to a compact spatial summary of DINO
    while every DINO patch attends to valid text tokens.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 8,
        pool_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("reciprocal hidden_dim must be divisible by num_heads")
        if pool_size <= 0:
            raise ValueError("reciprocal pool_size must be positive")

        self.pool_size = int(pool_size)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        self.visual_query = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.text_key = nn.Linear(text_dim, hidden_dim, bias=False)
        self.text_value = nn.Linear(text_dim, hidden_dim, bias=False)
        self.text_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.text_query = nn.Linear(text_dim, hidden_dim, bias=False)
        self.visual_key = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.visual_value = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.visual_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.local_visual = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )
        self.visual_output = nn.Linear(hidden_dim, visual_dim, bias=False)
        self.text_output = nn.Linear(hidden_dim, text_dim, bias=False)
        self.visual_gate = nn.Linear(text_dim, visual_dim)
        self.text_gate = nn.Linear(visual_dim, text_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.visual_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.text_scale_logit = nn.Parameter(torch.tensor(-3.0))

        # Start from the frozen pretrained representation and learn the bridge
        # smoothly. V1 reciprocal checkpoints are intentionally new.
        nn.init.zeros_(self.visual_output.weight)
        nn.init.zeros_(self.text_output.weight)
        nn.init.zeros_(self.visual_gate.weight)
        nn.init.zeros_(self.visual_gate.bias)
        nn.init.zeros_(self.text_gate.weight)
        nn.init.zeros_(self.text_gate.bias)

    @staticmethod
    def _masked_mean(tokens: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask.bool()).unsqueeze(-1).to(tokens.dtype)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _to_map(tokens: Tensor, height: int, width: int) -> Tensor:
        if tokens.shape[1] != height * width:
            raise RuntimeError(
                f"visual token count {tokens.shape[1]} does not match {height}x{width}"
            )
        return tokens.reshape(
            tokens.shape[0], height, width, tokens.shape[-1]
        ).permute(0, 3, 1, 2)

    def forward(
        self,
        visual_tokens: Tensor,
        text_tokens: Tensor,
        text_padding_mask: Tensor,
        visual_grid: Tuple[int, int],
        special_tokens: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        height, width = visual_grid
        visual_special = visual_tokens[:, :special_tokens]
        visual_patches = visual_tokens[:, special_tokens:]
        normalized_visual = self.visual_norm(visual_patches).float()
        normalized_text = self.text_norm(text_tokens).float()

        # CLIP -> DINO: dense language grounding, excluding padding tokens.
        visual_query = self.visual_query(normalized_visual)
        text_update = self.text_to_visual(
            visual_query,
            self.text_key(normalized_text),
            self.text_value(normalized_text),
            key_padding_mask=text_padding_mask.bool(),
            need_weights=False,
        )[0]
        local_update = self.local_visual(
            self._to_map(visual_query, height, width)
        ).flatten(2).transpose(1, 2)
        visual_update = text_update + local_update

        # DINO -> CLIP: pool the grid before attention to bound memory at 448px.
        pooled_visual = F.adaptive_avg_pool2d(
            self._to_map(normalized_visual, height, width),
            output_size=(min(self.pool_size, height), min(self.pool_size, width)),
        ).flatten(2).transpose(1, 2)
        language_update = self.visual_to_text(
            self.text_query(normalized_text),
            self.visual_key(pooled_visual),
            self.visual_value(pooled_visual),
            need_weights=False,
        )[0]

        text_context = self._masked_mean(normalized_text, text_padding_mask)
        visual_context = pooled_visual.mean(dim=1)
        visual_gate = torch.sigmoid(self.visual_gate(text_context)).unsqueeze(1)
        text_gate = torch.sigmoid(self.text_gate(visual_context)).unsqueeze(1)

        visual_patches = visual_patches + torch.sigmoid(
            self.visual_scale_logit
        ) * visual_gate * self.visual_output(self.dropout(visual_update)).to(
            visual_patches.dtype
        )
        text_tokens = text_tokens + torch.sigmoid(
            self.text_scale_logit
        ) * text_gate * self.text_output(self.dropout(language_update)).to(
            text_tokens.dtype
        )
        text_tokens = text_tokens.masked_fill(
            text_padding_mask.unsqueeze(-1), 0.0
        )
        return torch.cat((visual_special, visual_patches), dim=1), text_tokens
