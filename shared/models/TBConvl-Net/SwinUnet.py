import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================ Separable Convolution =================================

class SeparableConv2d(nn.Module):
    """
    Implements a separable convolution layer using depthwise and pointwise convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# ================================== ConvLSTM2D ========================================

class ConvLSTMCell(nn.Module):
    """
    Implements a ConvLSTM cell.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        padding = kernel_size // 2  # To maintain spatial dimensions
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # along channel axis

        # Compute all gates at once
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)   # input gate
        f = torch.sigmoid(cc_f)   # forget gate
        o = torch.sigmoid(cc_o)   # output gate
        g = torch.tanh(cc_g)      # gate gate

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))

class ConvLSTM2D(nn.Module):
    """
    Implements a ConvLSTM2D layer that processes a sequence of inputs.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True, num_layers=1):
        super(ConvLSTM2D, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        layers = []
        for i in range(num_layers):
            input_c = input_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(input_c, hidden_channels, kernel_size, bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, input_tensor, reverse=False):
        # input_tensor shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = input_tensor.size()
        device = input_tensor.device

        # Initialize hidden and cell states for all layers
        hidden_state = []
        cell_state = []
        for i in range(self.num_layers):
            h, c = self.layers[i].init_hidden(batch_size, (height, width), device)
            hidden_state.append(h)
            cell_state.append(c)

        # Iterate over time steps
        if reverse:
            time_steps = reversed(range(seq_len))
        else:
            time_steps = range(seq_len)

        outputs = []
        for t in time_steps:
            x = input_tensor[:, t, :, :, :]  # (batch, channels, height, width)
            for i, layer in enumerate(self.layers):
                h, c = layer(x, (hidden_state[i], cell_state[i]))
                hidden_state[i] = h
                cell_state[i] = c
                x = h  # input to next layer
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, channels, height, width)
        if reverse:
            outputs = outputs.flip(dims=[1])  # Reverse back to original order
        return outputs  # Return the sequence of outputs

class BidirectionalConvLSTM2D(nn.Module):
    """
    Implements a Bidirectional ConvLSTM2D layer.
    Processes the input sequence in both forward and backward directions and concatenates the outputs.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=1, bias=True):
        super(BidirectionalConvLSTM2D, self).__init__()
        self.forward_conv_lstm = ConvLSTM2D(input_channels, hidden_channels, kernel_size, bias=bias, num_layers=num_layers)
        self.backward_conv_lstm = ConvLSTM2D(input_channels, hidden_channels, kernel_size, bias=bias, num_layers=num_layers)

    def forward(self, input_tensor):
        # input_tensor shape: (batch, seq_len, channels, height, width)
        # Forward direction
        forward_output = self.forward_conv_lstm(input_tensor, reverse=False)  # (batch, seq_len, hidden_channels, H, W)
        # Backward direction
        backward_output = self.backward_conv_lstm(input_tensor, reverse=True)  # (batch, seq_len, hidden_channels, H, W)
        # Concatenate outputs along the channel dimension
        output = torch.cat([forward_output, backward_output], dim=2)  # (batch, seq_len, hidden_channels*2, H, W)
        # Since seq_len=2, we can take the last output
        output = output[:, -1, :, :, :]  # Take the last output (batch, hidden_channels*2, H, W)
        return output

# ============================== Swin Transformer Blocks ================================

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim (int): Number of input channels.
            window_size (tuple): Height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            attn_drop (float): Dropout ratio of attention weights.
            proj_drop (float): Dropout ratio of output.
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Query, Key, Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, Wh*Ww, C)
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B_, nH, N, C//nH
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (B_, nH, N, C//nH)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # (B_, nH, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with W-MSA and SW-MSA.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # W
        self.shift_size = shift_size    # S
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads, qkv_bias, attn_drop, proj_drop)

        self.drop_path = nn.Identity()  # Can implement stochastic depth if desired
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        """
        Args:
            x: input features with shape (B, H*W, C)
        """
        H = W = int(np.sqrt(x.shape[1]))
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        window_size = self.window_size
        # Pad H and W to be multiples of window_size
        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        shifted_x = F.pad(shifted_x, (0, 0, 0, pad_r, 0, pad_b))  # pad H and W
        _, Hp, Wp, _ = shifted_x.shape

        # Window partition
        x_windows = shifted_x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)  # (num_windows*B, window_size*window_size, C)

        # Attention
        attn_windows = self.attn(x_windows)  # (num_windows*B, window_size*window_size, C)

        # Merge windows
        shifted_x = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = shifted_x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        x = x[:, :H, :W, :].contiguous().view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# =============================== Dice Loss Function ====================================

class DiceLoss(nn.Module):
    """
    Dice Loss function to maximize the Dice coefficient.
    Suitable for binary segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted mask probabilities with shape (B, 1, H, W)
            y_true (torch.Tensor): Ground truth masks with shape (B, 1, H, W)
        Returns:
            torch.Tensor: Dice loss
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return 1 - dice

# ================================ Main Model ============================================

class SwinUNet(nn.Module):
    """
    Swin U-Net architecture for image segmentation with bidirectional ConvLSTM layers.
    """
    def __init__(self, input_channels=3, output_channels=1,
                 embed_dim=32, num_heads=[4, 8], window_size=4,
                 mlp_ratio=4., depth=2):
        super(SwinUNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Initial convolutional layers
        self.conv1 = SeparableConv2d(input_channels, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = SeparableConv2d(24, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x256 -> 128x128

        # First Swin Transformer Block
        self.swin_unet_E1 = SwinTransformerBlock(
            dim=24,  # Changed from embed_dim=32 to 24
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=window_size//2 if True else 0,
            mlp_ratio=mlp_ratio
        )

        # Second convolutional block
        self.conv3 = SeparableConv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = SeparableConv2d(48, 48, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64

        # Second Swin Transformer Block
        self.swin_unet_E2 = SwinTransformerBlock(
            dim=48,
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size=window_size//2 if True else 0,
            mlp_ratio=mlp_ratio
        )

        # Third convolutional block (Bottleneck)
        self.conv5 = SeparableConv2d(48, 96, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = SeparableConv2d(96, 96, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(96)
        self.drop5 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Bottleneck convolutions with dense connections
        self.conv7 = SeparableConv2d(96, 192, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = SeparableConv2d(192, 192, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(192)
        self.drop6_1 = nn.Dropout(0.5)

        self.conv9 = SeparableConv2d(192, 192, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(192)
        self.conv10 = SeparableConv2d(192, 192, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(192)
        self.drop6_2 = nn.Dropout(0.5)

        self.concat1 = nn.Sequential(
            SeparableConv2d(384, 192, kernel_size=3, padding=1),
            SeparableConv2d(192, 192, kernel_size=3, padding=1)
        )
        self.drop6_3 = nn.Dropout(0.5)

        # First Upsampling Block
        self.up1 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.bn_up1 = nn.BatchNorm2d(96)
        self.relu_up1 = nn.ReLU(inplace=True)
        self.bidirectional_convLSTM1 = BidirectionalConvLSTM2D(input_channels=96, hidden_channels=192, kernel_size=3, num_layers=1)
        self.swin_unet_D1 = SwinTransformerBlock(
            dim=192 * 2,  # Adjusted for bidirectional output
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=window_size//2 if True else 0,
            mlp_ratio=mlp_ratio
        )
        self.conv11 = SeparableConv2d(192 * 2, 48, kernel_size=3, padding=1)
        self.conv12 = SeparableConv2d(48, 48, kernel_size=3, padding=1)

        # Second Upsampling Block
        self.up2 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)  # 64x64 -> 128x128
        self.bn_up2 = nn.BatchNorm2d(48)
        self.relu_up2 = nn.ReLU(inplace=True)
        self.bidirectional_convLSTM2 = BidirectionalConvLSTM2D(input_channels=48, hidden_channels=96, kernel_size=3, num_layers=1)
        self.swin_unet_D2 = SwinTransformerBlock(
            dim=96 * 2,
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size=window_size//2 if True else 0,
            mlp_ratio=mlp_ratio
        )
        self.conv13 = SeparableConv2d(96 * 2, 24, kernel_size=3, padding=1)
        self.conv14 = SeparableConv2d(24, 24, kernel_size=3, padding=1)

        # Third Upsampling Block
        self.up3 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)  # 128x128 -> 256x256
        self.bn_up3 = nn.BatchNorm2d(24)
        self.relu_up3 = nn.ReLU(inplace=True)
        self.bidirectional_convLSTM3 = BidirectionalConvLSTM2D(input_channels=24, hidden_channels=48, kernel_size=3, num_layers=1)
        self.swin_unet_D3 = SwinTransformerBlock(
            dim=48 * 2,
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size=window_size//2 if True else 0,
            mlp_ratio=mlp_ratio
        )
        self.conv15 = SeparableConv2d(48 * 2, 24, kernel_size=3, padding=1)
        self.conv16 = SeparableConv2d(24, 24, kernel_size=3, padding=1)

        # Output Layer
        self.final_conv1 = nn.Conv2d(24, 2, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Swin U-Net model.
        Args:
            x: Input tensor with shape (B, 3, 256, 256)
        Returns:
            torch.Tensor: Output segmentation mask with shape (B, 1, 256, 256)
        """
        # Initial Convolutions
        x1 = self.conv1(x)          # (B, 24, 256, 256)
        x1 = self.bn1(x1)
        x1 = self.conv2(x1)         # (B, 24, 256, 256)
        x1 = self.bn2(x1)
        p1 = self.pool1(x1)         # (B, 24, 128, 128)

        # First Swin Transformer Block
        p1_flat = p1.flatten(2).transpose(1, 2)  # (B, 128*128, 24)
        swin_E1 = self.swin_unet_E1(p1_flat)     # (B, 128*128, 24)
        swin_E1 = swin_E1.transpose(1, 2).view(-1, 24, 128, 128)  # Reshape for Conv2d

        # Second Convolutional Block
        x2 = self.conv3(swin_E1)    # (B, 48, 128, 128)
        x2 = self.bn3(x2)
        x2 = self.conv4(x2)          # (B, 48, 128, 128)
        x2 = self.bn4(x2)
        p2 = self.pool2(x2)          # (B, 48, 64, 64)

        # Second Swin Transformer Block
        p2_flat = p2.flatten(2).transpose(1, 2)  # (B, 64*64, 48)
        swin_E2 = self.swin_unet_E2(p2_flat)     # (B, 64*64, 48)
        swin_E2 = swin_E2.transpose(1, 2).view(-1, 48, 64, 64)  # Reshape for Conv2d

        # Third Convolutional Block (Bottleneck)
        x3 = self.conv5(swin_E2)    # (B, 96, 64, 64)
        x3 = self.bn5(x3)
        x3 = self.conv6(x3)          # (B, 96, 64, 64)
        x3 = self.bn6(x3)
        x3 = self.drop5(x3)
        p3 = self.pool3(x3)          # (B, 96, 32, 32)

        # Bottleneck Convolutions with Dense Connections
        x4 = self.conv7(p3)          # (B, 192, 32, 32)
        x4 = self.bn7(x4)
        x4 = self.conv8(x4)          # (B, 192, 32, 32)
        x4 = self.bn8(x4)
        x4 = self.drop6_1(x4)

        x5 = self.conv9(x4)          # (B, 192, 32, 32)
        x5 = self.bn9(x5)
        x5 = self.conv10(x5)         # (B, 192, 32, 32)
        x5 = self.bn10(x5)
        x5 = self.drop6_2(x5)

        concat = torch.cat([x5, x4], dim=1)  # (B, 384, 32, 32)
        concat = self.concat1(concat)         # (B, 192, 32, 32)
        concat = self.drop6_3(concat)         # (B, 192, 32, 32)

        # First Upsampling Block
        up1 = self.up1(concat)                 # (B, 96, 64, 64)
        up1 = self.bn_up1(up1)
        up1 = self.relu_up1(up1)

        # Prepare for BidirectionalConvLSTM2D
        up1_seq = torch.stack([x3, up1], dim=1)  # (B, 2, 96, 64, 64)
        bidir_convLSTM1_out = self.bidirectional_convLSTM1(up1_seq)  # (B, 192*2, 64, 64)

        # Swin Transformer Block in Decoder
        bidir_convLSTM1_flat = bidir_convLSTM1_out.flatten(2).transpose(1, 2)  # (B, 64*64, 192*2)
        swin_D1 = self.swin_unet_D1(bidir_convLSTM1_flat)               # (B, 64*64, 192*2)
        swin_D1 = swin_D1.transpose(1, 2).view(-1, 192*2, 64, 64)    # Reshape for Conv2d

        # Further Convolutions
        conv6 = self.conv11(swin_D1)        # (B, 48, 64, 64)
        conv6 = self.conv12(conv6)          # (B, 48, 64, 64)

        # Second Upsampling Block
        up2 = self.up2(conv6)               # (B, 48, 128, 128)
        up2 = self.bn_up2(up2)
        up2 = self.relu_up2(up2)

        # Prepare for BidirectionalConvLSTM2D
        up2_seq = torch.stack([x2, up2], dim=1)  # (B, 2, 48, 128, 128)
        bidir_convLSTM2_out = self.bidirectional_convLSTM2(up2_seq)  # (B, 96*2, 128, 128)

        # Swin Transformer Block in Decoder
        bidir_convLSTM2_flat = bidir_convLSTM2_out.flatten(2).transpose(1, 2)  # (B, 128*128, 96*2)
        swin_D2 = self.swin_unet_D2(bidir_convLSTM2_flat)               # (B, 128*128, 96*2)
        swin_D2 = swin_D2.transpose(1, 2).view(-1, 96*2, 128, 128)    # Reshape for Conv2d

        # Further Convolutions
        conv7 = self.conv13(swin_D2)        # (B, 24, 128, 128)
        conv7 = self.conv14(conv7)          # (B, 24, 128, 128)

        # Third Upsampling Block
        up3 = self.up3(conv7)               # (B, 24, 256, 256)
        up3 = self.bn_up3(up3)
        up3 = self.relu_up3(up3)

        # Prepare for BidirectionalConvLSTM2D
        up3_seq = torch.stack([x1, up3], dim=1)  # (B, 2, 24, 256, 256)
        bidir_convLSTM3_out = self.bidirectional_convLSTM3(up3_seq)  # (B, 48*2, 256, 256)

        # Swin Transformer Block in Decoder
        bidir_convLSTM3_flat = bidir_convLSTM3_out.flatten(2).transpose(1, 2)  # (B, 256*256, 48*2)
        swin_D3 = self.swin_unet_D3(bidir_convLSTM3_flat)               # (B, 256*256, 48*2)
        swin_D3 = swin_D3.transpose(1, 2).view(-1, 48*2, 256, 256)    # Reshape for Conv2d

        # Further Convolutions
        conv8 = self.conv15(swin_D3)        # (B, 24, 256, 256)
        conv8 = self.conv16(conv8)          # (B, 24, 256, 256)

        # Final Output Convolutions
        final = self.final_conv1(conv8)      # (B, 2, 256, 256)
        final = self.final_relu(final)
        final = self.final_conv2(final)      # (B, 1, 256, 256)
        final = self.final_sigmoid(final)    # (B, 1, 256, 256)

        return final

# ================================== Dataset Class ======================================

class SegmentationDataset(Dataset):
    """
    Custom Dataset for image segmentation tasks.
    Expects images in 'x' folder and masks in 'y' folder.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        super(SegmentationDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')    # Grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

