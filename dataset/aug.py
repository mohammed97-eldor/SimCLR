import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import crop, resize
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from data_generator import ContrastiveLearningDataset


CROP_PROPORTION = 0.875  # Standard for ImageNet.

def random_apply(func, p, x):
    """
    Randomly apply function func to x with probability p.
    
    Args:
        func (callable): Function to be applied.
        p (float): Probability of applying the function.
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor (with func applied or unchanged).
    """
    if random.random() < p:
        return func(x)
    return x

def random_brightness(image, max_delta):
    """
    Apply random brightness adjustment using SimCLRv2 method.
    
    Args:
        image (torch.Tensor): Input image tensor.
        max_delta (float): Maximum brightness change factor.

    Returns:
        torch.Tensor: Brightness-adjusted image.
    """
    # Randomly sample the multiplicative factor
    factor = torch.empty(1).uniform_(max(1.0 - max_delta, 0), 1.0 + max_delta).item()
    
    # Apply the factor to adjust brightness
    image = image * factor
    
    # Ensure values are clamped to valid range [0, 1] or [0, 255]
    image = torch.clamp(image, 0.0, 1.0 if image.max() <= 1.0 else 255.0)
    
    return image

def to_grayscale(image, keep_channels=True):
    """
    Convert an RGB image to grayscale.
    
    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W) where C=3 for RGB.
        keep_channels (bool): Whether to replicate grayscale values to 3 channels.
    
    Returns:
        torch.Tensor: Grayscale image tensor.
    """
    if image.shape[0] == 0:
        print("Images are already grayscale")
        return image

    if image.shape[0] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    
    # Convert to grayscale using weighted sum (luminosity method)
    grayscale = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
    
    if keep_channels:
        # Replicate the grayscale image across 3 channels
        grayscale = grayscale.unsqueeze(0).repeat(3, 1, 1)
    
    return grayscale

def color_jitter_rand(image, brightness, contrast, saturation, hue):
    """
    Apply color jitter with randomized order.
    """
    transforms = []
    
    if brightness > 0:
        transforms.append(lambda img: random_brightness(img, brightness))
    
    if contrast > 0:
        transforms.append(T.ColorJitter(contrast=contrast))
    
    if saturation > 0:
        transforms.append(T.ColorJitter(saturation=saturation))
    
    if hue > 0:
        transforms.append(T.ColorJitter(hue=hue))
    
    # Shuffle the transformations
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())  # Seed to ensure randomness for reproducibility
    np.random.shuffle(transforms)

    for transform in transforms:
        image = transform(image)
    
    return image

def color_jitter_nonrand(image, brightness, contrast, saturation, hue):
    """
    Apply color jitter in a fixed order.
    """
    if brightness > 0:
        image = random_brightness(image, brightness)
    
    if contrast > 0:
        image = T.ColorJitter(contrast=contrast)(image)
    
    if saturation > 0:
        image = T.ColorJitter(saturation=saturation)(image)
    
    if hue > 0:
        image = T.ColorJitter(hue=hue)(image)
    
    return image

def color_jitter(image, strength, random_order=True):
    """
    Distorts the color of the image.
    
    Args:
        image (torch.Tensor): The input image tensor.
        strength (float): The strength of the color augmentation.
        random_order (bool): Whether to randomize the jittering order.
        impl (str): 'simclrv1' or 'simclrv2' to determine the version of random brightness.
    
    Returns:
        torch.Tensor: The distorted image tensor.
    """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength

    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue)
    else:
        return color_jitter_nonrand(image, brightness, contrast, saturation, hue)

def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.

    Args:
        image: Image Tensor to crop.
        height: Height of image to be cropped.
        width: Width of image to be cropped.
        crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
        A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    img_height, img_width = image.shape[1], image.shape[2]  # [C, H, W] shape
    crop_height, crop_width = _compute_crop_shape(img_height, img_width, width / height, crop_proportion)
    
    # Calculate offset for cropping
    offset_height = (img_height - crop_height) // 2
    offset_width = (img_width - crop_width) // 2
    
    # Crop the image using slicing
    image = image[:, offset_height:offset_height + crop_height, offset_width:offset_width + crop_width]

    # Resize the cropped image
    image = F.interpolate(image.unsqueeze(0), size=(height, width), mode='bicubic', align_corners=False).squeeze(0)
    
    return image

def _compute_crop_shape(image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.

    Args:
        image_height: Height of image to be cropped.
        image_width: Width of image to be cropped.
        aspect_ratio: Desired aspect ratio (width / height) of output.
        crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
        crop_height: Height of image after cropping.
        crop_width: Width of image after cropping.
    """
    image_width_float = float(image_width)
    image_height_float = float(image_height)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = int(round(crop_proportion / aspect_ratio * image_width_float))
        crop_width = int(round(crop_proportion * image_width_float))
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = int(round(crop_proportion * image_height_float))
        crop_width = int(round(crop_proportion * aspect_ratio * image_height_float))
        return crop_height, crop_width

    if aspect_ratio > image_width_float / image_height_float:
        return _requested_aspect_ratio_wider_than_image()
    else:
        return _image_wider_than_requested_aspect_ratio()

def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0), max_attempts=100):
    """
    Generates a randomly distorted bounding box crop of the image.
    """
    _, img_height, img_width = image.shape

    for _ in range(max_attempts):
        target_area = torch.empty(1).uniform_(area_range[0], area_range[1]).item() * img_height * img_width
        aspect_ratio = torch.empty(1).uniform_(*aspect_ratio_range).item()
        crop_width = int(round((target_area * aspect_ratio) ** 0.5))
        crop_height = int(round((target_area / aspect_ratio) ** 0.5))

        if crop_width <= img_width and crop_height <= img_height:
            offset_height = torch.randint(0, img_height - crop_height + 1, (1,)).item()
            offset_width = torch.randint(0, img_width - crop_width + 1, (1,)).item()

            cropped_image = crop(image, offset_height, offset_width, crop_height, crop_width)
            return cropped_image

    # Fallback to center crop if no valid crop found
    return center_crop(image, img_height, img_width, crop_proportion=1.0)

def crop_and_resize(image, height, width):
    """
    Makes a random crop and resizes it to the desired dimensions.
    """
    bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # Simulates bounding box
    aspect_ratio = width / height
    cropped_image = distorted_bounding_box_crop(
        image,
        bbox=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100
    )
    resized_image = resize(cropped_image, (height, width), interpolation=InterpolationMode.BICUBIC)
    return resized_image

def gaussian_blur(image, kernel_size, sigma, padding='same'):
    """
    Blurs the given image with separable convolution.

    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W] to blur.
        kernel_size: Integer specifying the size of the blur kernel (odd number).
        sigma: Sigma value for the Gaussian operator.
        padding: Padding mode, either 'same' or 'valid'.

    Returns:
        A Tensor representing the blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel_size is odd
    radius = kernel_size // 2

    # Create the 1D Gaussian kernel
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    blur_filter = torch.exp(-x**2 / (2 * sigma**2))
    blur_filter /= blur_filter.sum()  # Normalize the kernel

    # Create vertical and horizontal filters
    blur_v = blur_filter.view(kernel_size, 1).unsqueeze(0)  # [1, kernel_size, 1]
    blur_h = blur_filter.view(1, kernel_size).unsqueeze(0)  # [1, 1, kernel_size]

    # Match the number of channels
    num_channels = image.shape[1] if image.dim() == 4 else image.shape[0]
    blur_v = blur_v.repeat(num_channels, 1, 1, 1)  # [C, 1, kernel_size, 1]
    blur_h = blur_h.repeat(num_channels, 1, 1, 1)  # [C, 1, 1, kernel_size]

    # Add batch dimension if necessary
    expand_batch_dim = image.dim() == 3
    if expand_batch_dim:
        image = image.unsqueeze(0)  # Add batch dimension [B, C, H, W]

    # Apply depthwise convolution for vertical and horizontal blurring
    if padding == 'same':
        padding_mode = 'same'
    elif padding == 'valid':
        padding_mode = 'valid'
    else:
        raise ValueError("Padding must be 'same' or 'valid'.")

    blurred = F.conv2d(image, blur_h, stride=1, padding=padding_mode, groups=num_channels)
    blurred = F.conv2d(blurred, blur_v, stride=1, padding=padding_mode, groups=num_channels)

    # Remove batch dimension if it was added
    if expand_batch_dim:
        blurred = blurred.squeeze(0)

    return blurred

def batch_random_blur(images_list, height, width, blur_probability=0.5):
    """Apply efficient batch data transformations with random blur.
    
    Args:
        images_list: A list of image tensors of shape [C, H, W] or [B, C, H, W].
        height: The height of the image.
        width: The width of the image.
        blur_probability: Probability to apply the blur operator.
        
    Returns:
        A list of tensors with possibly blurred images.
    """
    
    def generate_selector(p, bsz):
        """Generates a selector mask with a certain probability."""
        selector = (torch.rand(bsz, 1, 1, 1) < p).float()
        return selector

    new_images_list = []
    for images in images_list:
        # Apply random blur
        images_new = gaussian_blur(images, kernel_size=5, sigma=1.5)
        
        # Generate selector mask
        selector = generate_selector(blur_probability, images.shape[0])

        # Apply blur based on selector
        images = images_new * selector + images * (1 - selector)
        
        # Ensure pixel values are in [0, 1]
        images = torch.clamp(images, 0., 1.)
        
        # Append processed image to the list
        new_images_list.append(images)

    return new_images_list

def preprocess_for_train(image, height, width, color_distort=True, crop=True, flip=True):
    """Preprocesses the given image for training in PyTorch.
    
    Args:
        image: Tensor representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        color_distort: Whether to apply the color distortion.
        crop: Whether to crop the image.
        flip: Whether or not to flip left and right of an image.
        impl: 'simclrv1' or 'simclrv2'. Whether to use simclrv1 or simclrv2's version of random brightness.
        
    Returns:
        A preprocessed image Tensor.
    """
    
    # Step 1: Resize and crop the image
    if crop:
        image = crop_and_resize(image, height, width)
    
    # Step 2: Random horizontal flip
    if flip:
        image = T.functional.hflip(image)
    
    # Step 3: Random color jitter
    if color_distort:
        image = color_jitter(image, strength=0.5, random_order=True)
    
    # Step 4: Normalize and ensure correct shape
    image = image.permute(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
    image = torch.clamp(image, 0., 1.)  # Clip to ensure the pixel values are within [0, 1]
    
    return image

def main(path_download = "./data"):

    data_loader = ContrastiveLearningDataset(path_download).get_dataset()
    for batch in data_loader:
        break

    preprocessed_image = batch[0][0,:,:,:].permute(1,2,0).cpu().numpy()
    preprocessed_image = np.clip(preprocessed_image, 0, 1)
    plt.imshow(preprocessed_image)
    plt.axis('off')
    plt.title("Original Image")
    plt.show()

    preprocessed_image = preprocess_for_train(batch[0][0,:,:,:], height=32, width=32, color_distort=True, crop=True, flip=True)
    preprocessed_image = preprocessed_image.cpu().numpy()
    preprocessed_image = np.clip(preprocessed_image, 0, 1)
    plt.imshow(preprocessed_image)
    plt.axis('off')
    plt.title("First Randomly Augmentation")
    plt.show()

    preprocessed_image = preprocess_for_train(batch[0][0,:,:,:], height=32, width=32, color_distort=True, crop=True, flip=True)
    preprocessed_image = preprocessed_image.cpu().numpy()
    preprocessed_image = np.clip(preprocessed_image, 0, 1)
    plt.imshow(preprocessed_image)
    plt.axis('off')
    plt.title("Second Randomly Augmentation")
    plt.show()

if __name__ == "__main__":
    main()
