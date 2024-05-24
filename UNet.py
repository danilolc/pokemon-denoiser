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


class ConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding='same', stride=1, dilation=1, bias=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride, 
                    dilation=dilation,
                    bias=bias),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()

        self.residual = residual

        self.conv1 = ConvolutionBlock(in_c, in_c, kernel_size=1)
        self.conv2 = ConvolutionBlock(in_c, in_c, kernel_size=3, dilation=6)  # 3
        self.conv3 = ConvolutionBlock(in_c, in_c, kernel_size=3, dilation=12) # 6
        self.conv4 = ConvolutionBlock(in_c, in_c, kernel_size=3, dilation=18) # 12

        self.conv5 = nn.Sequential(
            nn.AvgPool2d(8),
            ConvolutionBlock(in_c, in_c, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        self.conv6 = ConvolutionBlock(5 * in_c, out_c, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x6 = self.conv6(x6)

        if self.residual:
            return x + x6
        
        return x6

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
            #DoubleConv(in_channels, in_channels, residual=True),
            ASPP(in_channels, out_channels),
        )

        self.emb1_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        self.emb2_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        self.att = SelfAttention(out_channels)

    def forward(self, x, t, u):
        x = self.maxpool_conv(x)
        emb1 = self.emb1_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        emb2 = self.emb2_layer(u)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.att(x + emb1 + emb2)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, emb_dim=ED):
        super().__init__()

        #self.up = TConv(in_channels, in_channels)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(skip_channels + in_channels, skip_channels + in_channels, residual=True),
            #DoubleConv(skip_channels + in_channels, skip_channels + in_channels, residual=True),
            ASPP(skip_channels + in_channels, out_channels),
        )
        self.emb1_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        self.emb2_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        self.att = SelfAttention(out_channels)

    def forward(self, x, skip_x, t, u):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb1 = self.emb1_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        emb2 = self.emb2_layer(u)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.att(x + emb1 + emb2)


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=ED):
        super().__init__()
        self.time_dim = time_dim

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=1),
            DoubleConv(16, 16, residual=True),
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
            DoubleConv(16, 16, residual=True),
            nn.Conv2d(16, c_out, kernel_size=1),
        )

    def unet_forward(self, x, t, u):
        
        x1 = self.enc(x)

        x2 = self.down1(x1, t, u)
        x3 = self.down2(x2, t, u)
        x4 = self.down3(x3, t, u)

        x4 = self.bot(x4)

        x = self.up1(x4, x3, t, u)
        x = self.up2( x, x2, t, u)
        x = self.up3( x, x1, t, u)

        return self.dec(x)
    
    def forward(self, x, t, u):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        u = u.unsqueeze(-1)
        u = pos_encoding(u, self.time_dim)

        return self.unet_forward(x, t, u)

class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=ED, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, u, y=None):
        t = t.unsqueeze(-1)
        t = pos_encoding(t, self.time_dim)

        u = u.unsqueeze(-1)
        u = pos_encoding(u, self.time_dim)

        if y is not None:
            u += self.label_emb(y)

        return self.unet_forward(x, t, u)