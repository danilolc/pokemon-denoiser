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
        self.ln1 = nn.LayerNorm([channels])
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ln2 = nn.LayerNorm([channels])

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
            return F.gelu(self.double_conv(x))


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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            DoubleConv(out_channels, out_channels, residual=True),
            DoubleConv(out_channels, out_channels, residual=True),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            TConv(in_channels, in_channels),
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, in_channels + skip_channels, residual=True),
            DoubleConv(in_channels + skip_channels, out_channels),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)

        return x

class ContourEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.step1 = nn.Sequential(
            DoubleConv(1, 16),

            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.GroupNorm(1, 32),
            nn.GELU(),
        )
        self.step2 = nn.Sequential(
            DoubleConv(32, 32),

            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
        )
        self.step3 = nn.Sequential(
            DoubleConv(64, 64),

            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
        )
        self.mha1 = MHA(32)
        self.mha2 = MHA(64)
        self.mha3 = MHA(128)

    def forward(self, c):
        c1 = self.step1(c)
        c1 = self.mha1(c1, c1, c1)

        c2 = self.step2(c1)
        c2 = self.mha2(c2, c2, c2)

        c3 = self.step3(c2)
        c3 = self.mha3(c3, c3, c3)

        return c1, c2, c3

class TimeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.encoders = nn.ModuleList()
        for i in [32, 64, 128, 64, 32]:
            self.encoders.append(nn.Sequential(
                #nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, i),
            ))

    def forward(self, t):
        return [enc(t)[:, :, None, None] for enc in self.encoders]


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=1),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
        )
        
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.bot = nn.Sequential(
            DoubleConv(128, 128),
            DoubleConv(128, 128),
        )

        self.up1 = Up(128, 64, 64)
        self.up2 = Up(64, 32, 32)
        self.up3 = Up(32, 16, 16)
        
        self.dec = nn.Sequential(
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            nn.Conv2d(16, c_out, kernel_size=1),
        )

        self.c_encoder = ContourEncoder()
        self.t_encoder = TimeEncoder(self.time_dim)

        self.att1c = MHA(32)
        self.att1x = MHA(32)

        self.att2c = MHA(64)
        self.att2x = MHA(64)
        
        self.att3c = MHA(128)
        self.att3x = MHA(128)

        self.att4c = MHA(64)
        self.att4x = MHA(64)

        self.att5c = MHA(32)
        self.att5x = MHA(32)


    def unet_forward(self, x, t, c):

        c1, c2, c3 = self.c_encoder(c)
        t1, t2, t3, t4, t5 = self.t_encoder(t)
        
        x1 = self.enc(x)
        
        x2 = self.down1(x1)
        x2 = self.att1c(x2 + c1, x2 + c1, x2 + c1)
        x2 = self.att1x(x2 + t1, x2 + t1, x2 + t1)

        x3 = self.down2(x2)
        x3 = self.att2c(x3 + c2, x3 + c2, x3 + c2)
        x3 = self.att2x(x3 + t2, x3 + t2, x3 + t2)

        x4 = self.down3(x3)
        x4 = self.att3c(x4 + c3, x4 + c3, x4 + c3)
        x4 = self.att3x(x4 + t3, x4 + t3, x4 + t3)

        x4 = self.bot(x4)

        x4 = self.up1(x4, x3)
        x4 = self.att4c(x4 + c2, x4 + c2, x4 + c2)
        x4 = self.att4x(x4 + t4, x4 + t4, x4 + t4)

        x4 = self.up2(x4, x2)
        x4 = self.att5c(x4 + c1, x4 + c1, x4 + c1)
        x4 = self.att5x(x4 + t5, x4 + t5, x4 + t5)

        x4 = self.up3(x4, x1)

        return self.dec(x4)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        return self.unet_forward(x, t)

class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=10, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y, c):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)
        #t += self.label_emb(y[:, 0])

        return self.unet_forward(x, t, c)