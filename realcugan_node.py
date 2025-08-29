import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import folder_paths
from PIL import Image

def q(inp, cache_mode):
    maxx = inp.max()
    minn = inp.min()
    delta = maxx - minn
    if cache_mode == 2:
        return ((inp - minn) / delta * 255).round().byte().cpu(), delta, minn, inp.device
    elif cache_mode == 1:
        return ((inp - minn) / delta * 255).round().byte(), delta, minn, inp.device

def dq(inp, if_half, cache_mode, delta, minn, device):
    if cache_mode == 2:
        if if_half:
            return inp.to(device).half() / 255 * delta + minn
        else:
            return inp.to(device).float() / 255 * delta + minn
    elif cache_mode == 1:
        if if_half:
            return inp.half() / 255 * delta + minn
        else:
            return inp.float() / 255 * delta + minn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if "Half" in x.type():
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z

class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x, alpha=1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        z = self.conv_bottom(x5) * alpha
        return z

class UpCunet3x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(UpCunet3x, self).__init__()
        self.unet1 = UNet1x3(in_nc, out_nc, deconv=True)
        self.unet2 = UNet2(in_nc, out_nc, deconv=False)

    def forward(self, x, tile_mode=0, alpha=1):
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 4 + 1) * 4
        pw = ((w0 - 1) // 4 + 1) * 4
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), 'reflect')
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x, alpha)
        x = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x)
        if w0 != pw or h0 != ph:
            x = x[:, :, :h0 * 3, :w0 * 3]
        return x

class UpCunet4x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(UpCunet4x, self).__init__()
        self.unet1 = UNet1x3(in_nc, out_nc, deconv=True)
        self.unet2 = UNet2(in_nc, out_nc, deconv=False)

    def forward(self, x, tile_mode=0, alpha=1):
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 8 + 1) * 8
        pw = ((w0 - 1) // 8 + 1) * 8
        x = F.pad(x, (6, 6 + pw - w0, 6, 6 + ph - h0), 'reflect')
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x, alpha)
        x = F.pad(x, (-8, -8, -8, -8))
        x = torch.add(x0, x)
        if w0 != pw or h0 != ph:
            x = x[:, :, :h0 * 4, :w0 * 4]
        return x

class UpCunet2x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, inf=4):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_nc, out_nc * inf, deconv=True)
        self.unet2 = UNet2(out_nc * inf, out_nc, deconv=False)

    def forward(self, x, tile_mode=0, alpha=1):
        n, c, h0, w0 = x.shape
        if "Half" in x.type():
            if_half = True
        else:
            if_half = False
        
        if tile_mode == 0:  # No tiling
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x, alpha)
            x = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x)
            if w0 != pw or h0 != ph:
                x = x[:, :, :h0 * 2, :w0 * 2]
            return x
        else:
            # Simplified tiling mode for ComfyUI
            return self.forward_simple_tile(x, alpha)
    
    def forward_simple_tile(self, x, alpha):
        # Simple tiling implementation
        n, c, h0, w0 = x.shape
        tile_size = 256
        overlap = 32
        
        # Calculate output size
        out_h, out_w = h0 * 2, w0 * 2
        result = torch.zeros((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
        
        for i in range(0, h0, tile_size - overlap):
            for j in range(0, w0, tile_size - overlap):
                # Extract tile
                tile = x[:, :, i:i+tile_size, j:j+tile_size]
                
                # Pad if needed
                pad_h = max(0, tile_size - tile.shape[2])
                pad_w = max(0, tile_size - tile.shape[3])
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), 'reflect')
                
                # Process tile
                with torch.no_grad():
                    ph = ((tile.shape[2] - 1) // 2 + 1) * 2
                    pw = ((tile.shape[3] - 1) // 2 + 1) * 2
                    tile = F.pad(tile, (18, 18 + pw - tile.shape[3], 18, 18 + ph - tile.shape[2]), 'reflect')
                    tile = self.unet1.forward(tile)
                    tile0 = self.unet2.forward(tile, alpha)
                    tile = F.pad(tile, (-20, -20, -20, -20))
                    tile = torch.add(tile0, tile)
                    
                    # Place in result
                    out_i, out_j = i * 2, j * 2
                    out_h_end = min(out_h, out_i + tile.shape[2])
                    out_w_end = min(out_w, out_j + tile.shape[3])
                    result[:, :, out_i:out_h_end, out_j:out_w_end] = tile[:, :, :out_h_end-out_i, :out_w_end-out_j]
        
        return result

class RealCUGANUpscaler:
    def __init__(self):
        self.models = {}
        self.model_dir = folder_paths.get_folder_paths("upscale_models")[0]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ([
                    "up2x-latest-conservative.pth",
                    "up2x-latest-denoise1x.pth", 
                    "up2x-latest-denoise2x.pth",
                    "up2x-latest-denoise3x.pth",
                    "up2x-latest-no-denoise.pth",
                    "up3x-latest-conservative.pth",
                    "up3x-latest-denoise3x.pth", 
                    "up3x-latest-no-denoise.pth",
                    "up4x-latest-conservative.pth",
                    "up4x-latest-denoise3x.pth",
                    "up4x-latest-no-denoise.pth"
                ],),
                "tile_mode": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def load_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_dir}")
        
        # Initialize model architecture based on filename
        if "up2x" in model_name:
            model = UpCunet2x()
        elif "up3x" in model_name:
            model = UpCunet3x()
        elif "up4x" in model_name:
            model = UpCunet4x()
        else:
            # Default to 2x
            model = UpCunet2x()
        
        # Load weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        self.models[model_name] = model
        return model
    
    def upscale(self, image, model_name, tile_mode, alpha):
        # Load model
        model = self.load_model(model_name)
        device = next(model.parameters()).device
        
        # Convert ComfyUI image format to tensor
        # ComfyUI images are in format [batch, height, width, channels] with values 0-1
        image_tensor = image.permute(0, 3, 1, 2).to(device)  # [batch, channels, height, width]
        
        # Process image
        with torch.no_grad():
            result = model(image_tensor, tile_mode=tile_mode, alpha=alpha)
            result = torch.clamp(result, 0, 1)
        
        # Convert back to ComfyUI format
        result = result.permute(0, 2, 3, 1).cpu()
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "RealCUGANUpscaler": RealCUGANUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealCUGANUpscaler": "Real-CUGAN Upscaler",
}
