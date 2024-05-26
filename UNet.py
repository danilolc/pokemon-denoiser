import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE

def pos_encoding(t, channels):
    log_inv_freq = (-9.21034 / channels) * torch.arange(0.0, channels, 2.0, device=t.device)
    pos_enc_a = torch.sin(t * torch.exp(log_inv_freq[None, :]))
    pos_enc_b = torch.cos(t * torch.exp(log_inv_freq[None, :]))
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

class MHA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ln = nn.LayerNorm([channels])

    def forward(self, q, k, v):
        size = v.shape[-2:]
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        att, _ = self.mha(q, k, v)
        att = self.ln1(att + v)
        att = att.transpose(2, 1).unflatten(2, size)

        att = att.transpose(1, -1)
        att = self.ln2(att + self.linear(att))
        att = att.transpose(-1, 1)

        return att

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
    def __init__(self, in_channels, out_channels, emb_dim, att=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.att = att
        if self.att:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, out_channels),
            )
            self.mha1 = MHA(out_channels)
            self.mha2 = MHA(out_channels)

    def forward(self, x, t):
        
        x = self.maxpool_conv(x)
        
        if self.att:
            emb = self.emb_layer(t)
            x = x + emb[:, :, None, None]

            x = self.mha1(x, x, x)
            x = self.mha2(x, x, x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, emb_dim, att=True):
        super().__init__()
        self.up = nn.Sequential(
            TConv(in_channels, in_channels),
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, in_channels + skip_channels, residual=True),
            DoubleConv(in_channels + skip_channels, out_channels),
        )

        self.att = att
        if self.att:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, out_channels),
            )
            self.mha1 = MHA(skip_channels)
            self.mha2 = MHA(skip_channels)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)
        
        if self.att:
            emb = self.emb_layer(t)
            x = x + emb[:, :, None, None]

            x = self.mha1(skip_x, skip_x, x)
            x = self.mha2(x, x, x)

        return x


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=1),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
        )
        
        self.down1 = Down(16, 32, time_dim, att=False)
        self.down2 = Down(32, 64, time_dim)
        self.down3 = Down(64, 128, time_dim)

        self.bot = nn.Sequential(
            DoubleConv(128, 128),
            DoubleConv(128, 128),
        )

        self.up1 = Up(128, 64, 64, time_dim)
        self.up2 = Up(64, 32, 32, time_dim)
        self.up3 = Up(32, 16, 16, time_dim, att=False)
        
        self.dec = nn.Sequential(
            DoubleConv(16, 16, residual=True),
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
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y).sum(dim=1)

        return self.unet_forward(x, t)