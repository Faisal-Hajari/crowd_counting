from typing import Any, Union, Tuple
import torch 
from torchvision import transforms

#lazy work:: 
from torchvision.transforms import Compose, RandomApply, RandomChoice, RandomOrder, InterpolationMode

__all__ = [
    "Compose",#
    "RandomApply",#
    "RandomChoice",#
    "RandomOrder",#
    "InterpolationMode",#

    "Normalize",#
    "ColorJitter",#
    "RandomGrayscale",#
    "Grayscale", #
    "GaussianBlur",#
    "RandomInvert",#
    "RandomPosterize",#
    "RandomSolarize",#
    "RandomAdjustSharpness",#
    "RandomAutocontrast",#
    "RandomEqualize",#
    "ElasticTransform",#

    "Resize",
    "CenterCrop",
    "Pad",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResizedCrop",
    "FiveCrop",
    "TenCrop",
    "LinearTransformation",
    "RandomRotation",
    "RandomAffine",
    "RandomPerspective",
    "RandomErasing",
]
"""TODO: 
    1- General Crop function that can keep the labels consistant. 
    2- General affine function that can keep the labels consistant. if not: 
    3- General Rotation function that can keep the labels consistant. 
    4- General flip function that can keep the labels consistant
    5- General Erasing function that can keep the labels consistant
    6- General Perspective function that can keep the labels consistant
"""

"""here we'll seperate affine transforms (functions that can change the position of the object in the image, like scaling, cropps, and rotations)
and non-affine transforms (functions that don't change the position of the object in the image, like blurr, and normlization, and collour shuffling)"""

class NonAffineTransforms(torch.nn.Module): 
    """Base class for all non-affine transforms. e.g. transform.Blur()"""
    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, data) -> Any:
        data['image'] =  self.transform(data["image"])
        return data
    
    def __repr__(self) -> str:
        try: 
            return self.transform.__repr__()
        except: 
            return self.__class__.__name__

class ColorJitter(NonAffineTransforms): 
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """
    def __init__(self,         
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
        ) -> None:
        super().__init__(transforms.ColorJitter(brightness, contrast, saturation, hue))

class RandomGrayscale(NonAffineTransforms): 
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """
    def __init__(self, p=0.1) -> None:
        super().__init__(transforms.RandomGrayscale(p))

class Grayscale(RandomGrayscale): 
    """Convert image to grayscale.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Returns:
        PIL Image or Tensor: Grayscale version of the input image.
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """
    def __init__(self) -> None:
        super().__init__(p=1)

class Normalize(NonAffineTransforms):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__(transforms.Normalize(mean, std, inplace))

class GaussianBlur(NonAffineTransforms):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means at most one leading dimension.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """
    def __init__(self, kernel_size, sigma=(0.1, 2.0)) -> None:
        super().__init__(transforms.GaussianBlur(kernel_size, sigma))

class RandomInvert(NonAffineTransforms):
    """Inverts the colors of the given image randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """
    def __init__(self, p:float=0.5) -> None:
        super().__init__(transforms.RandomInvert(p))

class RandomPosterize(NonAffineTransforms):
    """Posterize the image randomly with a given probability by reducing the
    number of bits for each color channel. If the image is torch Tensor, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being posterized. Default value is 0.5
    """
    def __init__(self, bits:int, p:float=0.5) -> None:
        super().__init__(transforms.RandomPosterize(bits, p))

class RandomSolarize(NonAffineTransforms):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        threshold (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being solarized. Default value is 0.5
    """
    def __init__(self, threshold, p:float=0.5):
        super.__init__(transforms.RandomSolarize(threshold, p))

class RandomAdjustSharpness(NonAffineTransforms):
    """Adjust the sharpness of the image randomly with a given probability. If the image is torch Tensor,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p (float): probability of the image being sharpened. Default value is 0.5
    """

    def __init__(self, sharpness_factor, p=0.5):
        super().__init__(transforms.RandomAdjustSharpness(sharpness_factor, p))

class RandomAutocontrast(NonAffineTransforms):
    """Autocontrast the pixels of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """

    def __init__(self, p:float=0.5):
        super().__init__(transforms.RandomAutocontrast(p))

class RandomEqualize(torch.nn.Module):
    """Equalize the histogram of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Args:
        p (float): probability of the image being equalized. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(transforms.RandomEqualize(p))

class ElasticTransform(torch.nn.Module): 
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        alpha (float or sequence of floats): Magnitude of displacements. Default is 50.0.
        sigma (float or sequence of floats): Smoothness of displacements. Default is 5.0.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    .note: 
    this might affect the labels or considered an affine transformation. however, it's treated as nonAffine with the
    assumption that alpha is moderatly set to small number (i.e. < 15).
    """

    def __init__(self, alpha=50.0, sigma=5.0, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__(transforms.ElasticTransform(alpha, sigma, interpolation, fill))









