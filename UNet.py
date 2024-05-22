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

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


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
    def __init__(self, in_channels, out_channels, emb_dim=ED):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, emb_dim=ED):
        super().__init__()

        #self.up = TConv(in_channels, in_channels)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(skip_channels + in_channels, skip_channels + in_channels, residual=True),
            DoubleConv(skip_channels + in_channels, out_channels, in_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=ED):
        super().__init__()
        self.time_dim = time_dim

        self.inc = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=1),
            DoubleConv(16, 16),
        )
        
        self.down1 = Down(16, 32)
        self.sa1 = SelfAttention(32)
        self.down2 = Down(32, 64)
        self.sa2 = SelfAttention(64)
        self.down3 = Down(64, 128)
        self.sa3 = SelfAttention(128)

        self.bot = nn.Sequential(
            DoubleConv(128, 128, residual=True),
            DoubleConv(128, 128),
        )

        self.up1 = Up(128, 64, 64)
        self.sa4 = SelfAttention(64)
        self.up2 = Up(64, 32, 32)
        self.sa5 = SelfAttention(32)
        self.up3 = Up(32, 16, 16)
        self.sa6 = SelfAttention(16)
        
        self.outc = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, c_out, kernel_size=1),
        )



    def unet_forward(self, x, t):
        
        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.outc(x)
    
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
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)
