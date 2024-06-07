import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViT_Encoder(nn.Module):
    """
    适配Chromosome Patching的ViT encoder。

    """
    def __init__(self, args, use_classifier,
                 patch_dim=1024, dim=1024, depth=6, heads=16, dim_head=64, mlp_dim=1024, dropout=0.1, channel=3):
        super().__init__()
        self.args = args
        self.use_classifier = use_classifier
        self.encoding_size = dim

        self.embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.postional_embedding = nn.Parameter(torch.randn(size=(1, args.grid_size, dim)))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

        self.classifier = nn.Linear(dim, args.num_classes)

    def forward(self, x):  # (batch_size, grid_size(13), 1, 3, patch_size(32), patch_size)
        x = torch.mean(x, dim=3, keepdim=True)
        x = rearrange(x, 'b g 1 c p1 p2 -> b g (c p1 p2)')
        x = self.embedding(x)

        x = x + self.postional_embedding
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)
        x = x.unsqueeze(2)

        if self.use_classifier:
            x = x.view(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
            x = self.classifier(x)
            x = x.view(-1, self.args.grid_size * 1, self.args.num_classes)
            x = torch.mean(x, dim=1)

        return x


