import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE

def pos_encoding(t, channels):
    log_inv_freq = (-9.21034 / channels) * torch.arange(0.0, channels, 2.0, device=t.device)
    pos_enc_a = torch.sin(t[..., None] * torch.exp(log_inv_freq[None, :]))
    pos_enc_b = torch.cos(t[..., None] * torch.exp(log_inv_freq[None, :]))
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

#https://www.mdpi.com/2072-4292/11/9/1015/pdf
#@torch.compile
def ssim_loss(pred, target):
    mean_pred = torch.mean(pred, dim=(1, 2, 3))
    mean_target = torch.mean(target, dim=(1, 2, 3))

    var_pred = torch.var(pred, dim=(1, 2, 3))
    var_target = torch.var(target, dim=(1, 2, 3))

    cov = torch.mean(pred * target, dim=(1, 2, 3)) - mean_pred * mean_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (2 * mean_pred * mean_target + c1) * (2 * cov + c2) 
    ssim /= (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)

    return (1 - ssim) / 2

#https://www.researchgate.net/figure/The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the_fig2_348947034
class Transformer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mha = nn.MultiheadAttention(emb_dim, num_heads=2, batch_first=True, dropout=0.1)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        _, _, w, h = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, w * h, x.shape[1])

        x_ln = self.ln1(x)
        att, _ = self.mha(x_ln, x_ln, x_ln)
        
        att = self.ln2(att + x)
        att = self.mlp(att) + att

        return att.reshape(-1, w, h, att.shape[-1]).permute(0, 3, 1, 2)

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
            nn.GELU(),
        )

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)

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
            DoubleConv(out_channels, out_channels),
            DoubleConv(out_channels, out_channels, residual=True),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, out_channels),
            DoubleConv(out_channels, out_channels, residual=True),
        )

    def forward(self, x, skip_x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)

        return x

class ContourEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 64, 2, 2),
            Transformer(64),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 4),
            Transformer(128),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(1, 256, 8, 8),
            Transformer(256),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(1, 256, 8, 8),
            Transformer(256),
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 4),
            Transformer(128),
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(1, 64, 2, 2),
            Transformer(64),
        )

    def forward(self, c):
        return self.c1(c), self.c2(c), self.c3(c), self.c4(c), self.c5(c), self.c6(c)
                                                                                        
class TimeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoders = nn.ModuleList()
        for i in [64, 128, 256, 256, 128, 64]:
            self.encoders.append(nn.Linear(emb_dim, i))

    def forward(self, t, add = None):
        t = pos_encoding(t, self.emb_dim)
        if add is not None:
            t = t + add
        return [enc(t)[:, :, None, None] for enc in self.encoders]


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=10, time_dim=256):
        super().__init__()
        self.label_emb = nn.Embedding(18, time_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=1),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),
        )
        
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.bot = nn.Sequential(
            DoubleConv(256, 256),
            DoubleConv(256, 256),
        )

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)
        
        self.dec = nn.Sequential(
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),
            nn.Conv2d(32, c_out, kernel_size=1),
        )

        self.c_encoder = ContourEncoder()
        self.t_encoder = TimeEncoder(time_dim)

        self.att1 = Transformer(64)
        self.att2 = Transformer(128)
        self.att3 = Transformer(256)
        self.att4 = Transformer(256)
        self.att5 = Transformer(128)
        self.att6 = Transformer(64)

    def forward(self, x, t, ty, c):
        #c1, c2, c3, c4, c5, c6 = self.c_encoder(c)
        t1, t2, t3, t4, t5, t6 = self.t_encoder(t, self.label_emb(ty[:, 0]))
        
        x1 = self.enc(x)
        
        x2 = self.down1(x1)
        x2 = self.att1(x2 + t1)

        x3 = self.down2(x2)
        x3 = self.att2(x3 + t2)

        x4 = self.down3(x3)
        x4 = self.att3(x4 + t3)

        x4 = self.bot(x4)
        x4 = self.att4(x4 + t4)

        x4 = self.up1(x4, x3)
        x4 = self.att5(x4 + t5)

        x4 = self.up2(x4, x2)
        x4 = self.att6(x4 + t6)

        x4 = self.up3(x4, x1)

        return self.dec(x4)