import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE

ED = 256

def pos_encoding(t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device="cuda").float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

class MHA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, q, k, v):
        size = v.shape[-2:]
        q = q.flatten(2).swapaxes(1, 2)
        k = k.flatten(2).swapaxes(1, 2)
        v = v.flatten(2).swapaxes(1, 2)
        att, _ = self.mha(q, k, v)
        att = self.ln(att + v)

        return att.swapaxes(2, 1).unflatten(2, size)

class FF(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_c, in_c),
            nn.GELU(),
            nn.Linear(in_c, out_c),
        )
        self.ln = nn.LayerNorm([out_c])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x + self.linear(x))
        x = x.permute(0, 3, 1, 2)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class TConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.tconv = nn.Sequential(

            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.GroupNorm(1, out_c),
            nn.GELU(),

        )

    def forward(self, x):
        return self.tconv(x)

class Down(nn.Module):
    def __init__(self, img_sz, in_channels, out_channels, emb_dim=ED):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.vit = ViT(img_sz, img_sz // 8, out_channels)

    def forward(self, x, t):
        
        x = self.maxpool_conv(x)
        
        x = self.vit(x, t)

        return x


class Up(nn.Module):
    def __init__(self, img_sz, in_channels, skip_channels, out_channels, emb_dim=ED):
        super().__init__()
        self.up = nn.Sequential(
            TConv(in_channels, in_channels),
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, in_channels + skip_channels, residual=True),
            DoubleConv(in_channels + skip_channels, out_channels),
        )
        self.vit = ViT(img_sz, img_sz // 8, out_channels)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)

        x = self.vit(x, t)

        return x


class ViT_Embedding(nn.Module):
    def __init__(self, img_sz, patch_sz, channels, out_features):
        super().__init__()
        self.np = img_sz // patch_sz
        self.ps = patch_sz
        self.emb = nn.Linear(channels * self.ps * self.ps, out_features)
        self.pos_emb = nn.Parameter(
            torch.randn((self.np * self.np, out_features))
        )

    def forward(self, x):
        patches = x.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5) # [b, np, np, cc, ps, ps]

        sz = patches.shape[0:3]            # (b, np, np)
        patches = patches.flatten(0, 2)    # [b * np * np, cc, ps, ps]
        patches = patches.flatten(1, 3)    # [b * np * np, cc * ps * ps]
        patches = self.emb(patches)        # [b * np * np, out]
        patches = patches.unflatten(0, sz) # [b, np, np, out]

        patches = patches.flatten(1, 2)    # [b, np * np, out]

        return patches + self.pos_emb

class ViT(nn.Module):
    def __init__(self, img_sz, patch_sz, channels, time_dim=ED):
        super().__init__()
        self.cc = channels
        self.ps = patch_sz
        self.ddim = self.cc * self.ps * self.ps
        self.np = img_sz // patch_sz
        self.emb = ViT_Embedding(img_sz, patch_sz, self.cc, self.ddim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.ddim,
            nhead=4,
            dim_feedforward=self.ddim,
            batch_first=True
        )
        self.time_enc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                self.ddim
            ),
        )

    def forward(self, x, t):
        # x.shape -> [b, 3, img_sz, img_sz]

        x = self.emb(x)     # [b, np * np, 3 * ps * ps]
        x += self.time_enc(t)[:, None, :]
        x = self.encoder(x) # [b, np * np, 3 * ps * ps]

        x = x.unflatten(2, (self.cc, self.ps, self.ps)) # [b, np * np, 3, ps, ps]
        x = x.unflatten(1, (self.np, self.np))    # [b, np, np, 3, ps, ps]
        x = x.permute(0, 3, 1, 4, 2, 5)           # [b, 3, np, ps, np, ps]
        x = x.flatten(4, 5)                       # [b, 3, np, ps, img_sz]
        x = x.flatten(2, 3)                       # [b, 3, img_sz, img_sz]

        return x


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=ED):
        super().__init__()
        self.time_dim = time_dim

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=1),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
        )
        
        self.down1 = Down(36, 16, 32)
        self.down2 = Down(18, 32, 64)
        self.down3 = Down(9, 64, 128)

        self.bot = nn.Sequential(
            DoubleConv(128, 128),
            DoubleConv(128, 128),
        )

        self.up1 = Up(18, 128, 64, 64)
        self.up2 = Up(36, 64, 32, 32)
        self.up3 = Up(72, 32, 16, 16)
        
        self.dec = nn.Sequential(
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            nn.Conv2d(16, c_out, kernel_size=1),
        )

    def unet_forward(self, x, t):
        
        x1 = self.enc(x)

        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot(x4)

        x = self.up1(x4, x3, t)
        x = self.up2( x, x2, t)
        x = self.up3( x, x1, t)

        return self.dec(x)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        return self.unet_forward(x, t)

class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=ED, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y).sum(dim=1)

        return self.unet_forward(x, t)