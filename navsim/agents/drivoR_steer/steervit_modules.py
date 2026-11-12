"""SteerViT modules, copied from the official SteerViT code (MIT license).

Sources (unchanged bodies; parameter names are identical so the official
checkpoint loads with strict=True per module):
  - FeedForward, MaskedCrossAttention, GatedCrossAttention:
        SteerViT/src/steervit/crossattention.py
  - Connector:
        SteerViT/src/steervit/model.py
  - SoftCrossEntropyLoss:
        SteerViT/train.py

The only addition is ScaledGatedCrossAttention at the end of this file. It keeps
the parameters of GatedCrossAttention and adds (1) an external ramp factor s that
multiplies the gated residual, and (2) a per-row switch so rows without text return
their input unchanged (bit-exact).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def FeedForward(dim, mult):
    inner_dim = int(dim * mult)
    
    return nn.Sequential(
        nn.LayerNorm(dim), 
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class MaskedCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            head_dim,
            num_heads,
            ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        # for extraction
        self.attn_drop = nn.Dropout(0.0)
        self.attn_map = None
        self.save_attn = False  # set True when you want maps

    def forward(self, image_feats, text_feats, attn_mask = None):
        image_feats = self.norm(image_feats)

        if attn_mask is not None:
            attn_mask = attn_mask[:, image_feats.size(1):]      # (B, T_k)
            attn_mask = attn_mask[:, None, None, :]             # (B, 1, 1, T_k) broadcastable

        B, T_q, _ = image_feats.shape
        q = self.to_q(image_feats).reshape(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        T_k = text_feats.size(1)
        kv = self.to_kv(text_feats).reshape(B, T_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        if not self.save_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q * scale) @ k.transpose(-2, -1)  # (B, H, T_q, T_k)

            if attn_mask is not None:
                # attn_mask is boolean where True = keep (as in SDPA boolean masks)
                # convert to additive mask
                neg_inf = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(~attn_mask, neg_inf)

            attn = attn.softmax(dim=-1)
            self.attn_map = attn.detach()
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T_q, self.inner_dim)
        return self.to_out(out)

class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        layer_idx,
        dim,
        ff_mult,
        use_ffn,
        head_dim=72,
        num_heads=16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.cross_attn = MaskedCrossAttention(
            dim, 
            head_dim,
            num_heads,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.use_ffn = use_ffn
        if use_ffn:
            self.ff = FeedForward(dim, mult=ff_mult)
            self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, text_feats, attn_mask):
        
        attn_gate = self.attn_gate.tanh()
        x = x + attn_gate * (
            self.cross_attn(
                x,
                text_feats,
                attn_mask = attn_mask
            )
        )
            
        if self.use_ffn:        
            ff_gate = self.ff_gate.tanh()
            x = x + (ff_gate * self.ff(x))
        return x



class Connector(nn.Module):
    def __init__(self, input_dim, output_dim=1152):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets, reduction='mean', bbox = None):
        if not bbox:
            log_probs = F.log_softmax(logits, dim = -1)
            loss = -(targets * log_probs).sum(dim=-1)
            if reduction is None:
                return loss
            return loss.mean()



# ---------------------------------------------------------------------------
# Addition for Steer-DrivoR (not in SteerViT).
# ---------------------------------------------------------------------------
class ScaledGatedCrossAttention(GatedCrossAttention):
    """GatedCrossAttention with an external ramp factor and per-row skipping.

    Same parameters as GatedCrossAttention (cross_attn.*, attn_gate, ff.*, ff_gate).
    With gate_scale == 1 and every row active, the arithmetic equals the parent's
    forward. Rows with row_active == False return x unchanged (torch.where, bit-exact),
    so an empty-text camera reproduces the original DrivoR visual path.
    """

    def forward(self, x, text_feats, attn_mask, gate_scale=None, row_active=None):
        attn_gate = self.attn_gate.tanh()
        if gate_scale is not None:
            attn_gate = attn_gate * gate_scale
        y = x + attn_gate * (
            self.cross_attn(
                x,
                text_feats,
                attn_mask=attn_mask
            )
        )

        if self.use_ffn:
            ff_gate = self.ff_gate.tanh()
            if gate_scale is not None:
                ff_gate = ff_gate * gate_scale
            y = y + (ff_gate * self.ff(y))

        if row_active is not None:
            y = torch.where(row_active[:, None, None], y, x)
        return y
