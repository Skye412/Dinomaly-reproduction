"""Patch-only token adapter for the pinned SCRWKV AMCM; no added residual."""
import torch
from torch import nn
from models.amcm_official import AMCM


class AMCMTokenAdapter(nn.Module):
    def __init__(self, encoder, input_size=392):
        super().__init__()
        self.dim = encoder.embed_dim
        self.cls_tokens = int(encoder.cls_token.shape[1])
        self.register_tokens = int(encoder.num_register_tokens)
        patch_size = encoder.patch_size
        if isinstance(patch_size, (tuple, list)):
            assert patch_size[0] == patch_size[1]
            patch_size = patch_size[0]
        assert input_size % patch_size == 0 and self.dim == 768
        self.side = input_size // patch_size
        self.special_tokens = self.cls_tokens + self.register_tokens
        self.amcm = AMCM(dim=self.dim, sa_grid_size=5, sa_reduction=2,
                         multiscale_kernels=(5, 7, 9, 11))

    def forward(self, tokens):
        assert tokens.ndim == 3 and tokens.shape[2] == self.dim, 'AMCM token channel mismatch'
        batch, count, _ = tokens.shape
        assert count == self.special_tokens + self.side ** 2, 'AMCM patch grid mismatch'
        special = tokens[:, :self.special_tokens]
        patches = tokens[:, self.special_tokens:].transpose(1, 2).reshape(batch, self.dim, self.side, self.side)
        patches = self.amcm(patches)
        assert patches.shape == (batch, self.dim, self.side, self.side), 'AMCM changed grid shape'
        # The official module already returns out + residual. Never add tokens again.
        output = torch.cat((special, patches.flatten(2).transpose(1, 2)), dim=1)
        assert output.shape == tokens.shape
        return output
