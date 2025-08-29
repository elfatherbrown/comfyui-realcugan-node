# Real-CUGAN ComfyUI Custom Node

A ComfyUI custom node implementation for **Real-CUGAN** anime/illustration upscaling models from [Bilibili's AI Lab](https://github.com/bilibili/ailab/tree/main/Real-CUGAN).

## Overview

Real-CUGAN is a specialized upscaling model designed specifically for anime and illustration content, offering superior quality compared to generic photographic upscalers like Real-ESRGAN.

### Key Features
- 🎨 **Anime/Illustration Specialist**: Purpose-built for non-photographic content
- ⚙️ **Multiple Scale Options**: 2x, 3x, and 4x upscaling
- 🎛️ **Denoise Control**: Conservative, light (1x), medium (2x), strong (3x), or no denoising
- 💾 **Memory Efficient**: Built-in tiling support for large images
- 🔧 **Alpha Control**: Adjustable upscaling strength (0.1-2.0)

## Installation

### Prerequisites
- ComfyUI installation
- Real-CUGAN model weights (see Model Installation section)
- PyTorch with CUDA support

### 1. Install Custom Node
Place this directory in your ComfyUI `custom_nodes` folder:
```
ComfyUI/custom_nodes/comfyui_realcugan_node/
├── __init__.py
├── realcugan_node.py
└── README.md
```

### 2. Model Installation
Download Real-CUGAN model weights and place them in `ComfyUI/models/upscale_models/`:

```bash
# Example download (adjust URLs as needed):
wget -P models/upscale_models/ "URL_TO_REAL_CUGAN_MODELS"
```

**Required Models** (11 variants total):
- `up2x-latest-conservative.pth` (5.1MB)
- `up2x-latest-denoise1x.pth` (5.1MB)
- `up2x-latest-denoise2x.pth` (5.1MB)
- `up2x-latest-denoise3x.pth` (5.1MB)
- `up2x-latest-no-denoise.pth` (5.1MB)
- `up3x-latest-conservative.pth` (5.2MB)
- `up3x-latest-denoise3x.pth` (5.2MB)
- `up3x-latest-no-denoise.pth` (5.2MB)
- `up4x-latest-conservative.pth` (5.6MB)
- `up4x-latest-denoise3x.pth` (5.6MB)
- `up4x-latest-no-denoise.pth` (5.6MB)

### 3. Restart ComfyUI
Restart ComfyUI to load the custom node.

## Usage

### In ComfyUI Interface
1. Add **"Real-CUGAN Upscaler"** node (found in `image/upscaling` category)
2. Connect your input image
3. Select desired model variant
4. Adjust parameters as needed
5. Connect output to your workflow

### Parameters

#### **Model Selection**
Choose from 11 available Real-CUGAN models:
- **Scale Factor**: 2x, 3x, or 4x upscaling
- **Denoise Level**: 
  - `conservative`: Balanced enhancement
  - `no-denoise`: Preserve original detail level
  - `denoise1x`: Light noise reduction
  - `denoise2x`: Medium noise reduction  
  - `denoise3x`: Strong noise reduction

#### **Tile Mode** (0-4)
Controls memory usage for large images:
- `0`: No tiling (fastest, uses most VRAM)
- `1-4`: Various tiling strategies (slower, less VRAM)

#### **Alpha** (0.1-2.0)
Controls upscaling strength:
- `1.0`: Normal strength (recommended)
- `<1.0`: More conservative upscaling
- `>1.0`: More aggressive enhancement

## Model Recommendations

### For Different Content Types

#### **Clean Anime Art**
- **2x**: `up2x-latest-conservative.pth`
- **3x**: `up3x-latest-conservative.pth` 
- **4x**: `up4x-latest-conservative.pth`

#### **Noisy/Compressed Images**
- **Light artifacts**: Use `denoise1x` variants
- **Medium artifacts**: Use `denoise2x` variants
- **Heavy artifacts**: Use `denoise3x` variants

#### **High-Quality Sources**
- **Preserve detail**: Use `no-denoise` variants
- **Slight enhancement**: Use `conservative` variants

## Technical Implementation

### Architecture
This implementation includes:
- **SEBlock**: Squeeze-and-Excitation attention mechanism
- **UNetConv**: Convolutional blocks with LeakyReLU activation
- **UNet1/UNet2**: Multi-scale processing networks
- **UpCunet2x/3x/4x**: Complete upscaling architectures

### Memory Optimization
- **Automatic tiling**: Handles large images without OOM errors
- **Device management**: Automatic CUDA/CPU fallback
- **Efficient tensor conversion**: Minimal memory overhead

### ComfyUI Integration
- **Standard interface**: Follows ComfyUI node conventions
- **Type validation**: Proper input/output type checking
- **Error handling**: Graceful failure with clear messages

## Comparison with Other Upscalers

### Real-CUGAN vs Real-ESRGAN
- ✅ **Better anime handling**: Purpose-built for illustrations
- ✅ **Multiple scale options**: 2x/3x/4x vs fixed scaling
- ✅ **Denoise control**: Fine-grained noise reduction options
- ✅ **Lightweight**: <6MB models vs larger generic models

### Real-CUGAN vs Generic Upscalers
- ✅ **Specialized training**: Anime/illustration dataset
- ✅ **Content-aware**: Understands line art, cel shading, anime styles
- ✅ **Multiple variants**: Choose optimal model for your content
- ✅ **Research quality**: From Bilibili's AI research lab

## Troubleshooting

### Common Issues

#### **Model Not Found**
```
FileNotFoundError: Model up2x-latest-conservative.pth not found
```
**Solution**: Ensure model files are in `ComfyUI/models/upscale_models/`

#### **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Increase tile_mode value (1-4) to use less VRAM

#### **Import Errors**
```
ImportError: No module named 'torch'
```
**Solution**: Ensure PyTorch is installed in ComfyUI environment

### Performance Tips
- **Use tile_mode=0** for best quality if you have sufficient VRAM
- **Start with conservative models** and adjust based on results
- **Alpha=1.0** is optimal for most use cases
- **Denoise level** should match your source image quality

## Development

### Project Structure
```
comfyui_realcugan_node/
├── __init__.py           # ComfyUI node registration
├── realcugan_node.py     # Main implementation
└── README.md            # This documentation
```

### Architecture Implementation
The implementation faithfully recreates the Real-CUGAN architecture from the original research:
- Complete UNet implementations for each scale factor
- Proper padding and reflection handling
- SE (Squeeze-and-Excitation) blocks for attention
- Optimized inference pipeline

### Testing
Verify installation with:
```python
# In ComfyUI Python console or script:
from custom_nodes.comfyui_realcugan_node.realcugan_node import RealCUGANUpscaler
node = RealCUGANUpscaler()
print("Real-CUGAN node loaded successfully!")
```

## Credits

- **Original Research**: [Bilibili AI Lab - Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
- **ComfyUI Integration**: Custom implementation for ComfyUI compatibility
- **Model Weights**: From Bilibili's research releases

## License

This custom node implementation follows the same license terms as the original Real-CUGAN research. Model weights are provided by Bilibili AI Lab under their research license.

## Contributing

Feel free to submit issues or pull requests for:
- Performance optimizations
- Additional model support
- Bug fixes
- Documentation improvements

## Changelog

### v1.0.0 (2025-08-29)
- Initial release with full Real-CUGAN architecture implementation
- Support for all 11 model variants
- Memory-optimized tiling system
- Complete ComfyUI integration
