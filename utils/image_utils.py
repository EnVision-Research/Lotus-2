from PIL import Image
import matplotlib
import numpy as np
import cv2
from PIL import Image
import torch


def resize_image(image, target_size):
    """
    Resize output image to target size
    Args:
        image: Image in PIL.Image, numpy.array or torch.tensor format
        target_size: tuple, target size (H, W)
    Returns:
        Resized image in original format
    """
    if isinstance(image, list):
        return [resize_image(img, target_size) for img in image]
    
    if isinstance(image, Image.Image):
        return image.resize(target_size[::-1], Image.BILINEAR)
    elif isinstance(image, np.ndarray):
        # Handle numpy array with shape (1, H, W, 3)
        if image.ndim == 4:
            resized = np.stack([cv2.resize(img, target_size[::-1]) for img in image])
            return resized
        else:
            return cv2.resize(image, target_size[::-1])
    elif isinstance(image, torch.Tensor):
        # Handle tensor with shape (1, 3, H, W)
        if image.dim() == 4:
            return torch.nn.functional.interpolate(
                image,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            return torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=target_size, 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")

def resize_image_first(image_tensor, process_res=None):
    if process_res:
        max_edge = max(image_tensor.shape[2], image_tensor.shape[3])
        if max_edge > process_res:
            scale = process_res / max_edge
            new_height = int(image_tensor.shape[2] * scale)
            new_width = int(image_tensor.shape[3] * scale)
            image_tensor = resize_image(image_tensor, (new_height, new_width))
    
    image_tensor = resize_to_multiple_of_16(image_tensor)
    
    return image_tensor

def resize_to_multiple_of_16(image_tensor):
    """
    Resize image tensor to make shorter side closest multiple of 16 while maintaining aspect ratio
    Args:
        image_tensor: Input tensor of shape (B, C, H, W)
    Returns:
        Resized tensor where shorter side is multiple of 16
    """
    # Calculate scale ratio based on shorter side to make it closest multiple of 16
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    min_side = min(h, w)
    scale = (min_side // 16) * 16 / min_side
    
    # Calculate new height and width
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Ensure both height and width are multiples of 16
    new_h = (new_h // 16) * 16  
    new_w = (new_w // 16) * 16

    # Resize image while maintaining aspect ratio
    resized_tensor = torch.nn.functional.interpolate(
        image_tensor,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    return resized_tensor

def colorize_depth_map(depth, mask=None, reverse_color=False):
    cm = matplotlib.colormaps["Spectral"]
    # normalize
    depth = ((depth - depth.min()) / (depth.max() - depth.min()))
    # colorize
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]  # Invert the depth values before applying colormap
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3] # (h,w,3)

    depth_colored = (img_colored_np * 255).astype(np.uint8) 
    if mask is not None:
        masked_image = np.zeros_like(depth_colored)
        masked_image[mask.numpy()] = depth_colored[mask.numpy()]
        depth_colored_img = Image.fromarray(masked_image)
    else:
        depth_colored_img = Image.fromarray(depth_colored)
    return depth_colored_img

