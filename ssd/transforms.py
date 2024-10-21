import torch
import torchvision

import time
import pandas as pd
import csv
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional



def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

def get_image_name(target_id):
    tmp = target_id.split('[')
    tmp2 = tmp[1].split(']')
    image_id = tmp2[0]
    print("image id", image_id)
    ann_file = "/datasets/open-images-v6-mlperf/train/labels/openimages-mlperf.json"
    coco=COCO(ann_file)
    image_info = coco.loadImgs(int(image_id))
    return image_info[0]['file_name']

################################################################################
# TODO(ahmadki): remove this block, and replace get_image_size with F.get_image_size
#                once https://github.com/pytorch/vision/pull/4321 is public

from PIL import Image, ImageOps, ImageEnhance
Image.MAX_IMAGE_PIXELS = None
from typing import Any

try:
    import accimage
except ImportError:
    accimage = None


@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def get_image_size_tensor(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    # _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]

@torch.jit.unused
def get_image_size_pil(img: Any) -> List[int]:
    if _is_pil_image(img):
        return list(img.size)
    raise TypeError("Unexpected type {}".format(type(img)))

def get_image_size(img: Tensor) -> List[int]:
    """Returns the size of an image as [width, height].
    Args:
        img (PIL Image or Tensor): The image to be checked.
    Returns:
        List[int]: The image size.
    """
    if isinstance(img, torch.Tensor):
        return get_image_size_tensor(img)

    return get_image_size_pil(img)

def get_image_num_channels_tensor(img: Tensor) -> int:
    # _assert_image_tensor(img)
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError(f"Input ndim should be 2 or more. Got {img.ndim}")

@torch.jit.unused
def get_image_num_channels_pil(img: Any) -> int:
    if _is_pil_image(img):
        return len(img.getbands())
    raise TypeError("Unexpected type {}".format(type(img)))

def get_image_num_channels(img: Tensor) -> int:
    if isinstance(img, torch.Tensor):
        return get_image_num_channels_tensor(img)

    return get_image_num_channels_pil(img)
################################################################################

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
     
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, csv_file_path, p=0.5):
        super().__init__(p)  # Correctly initialize the parent class
        self.flip_times = []  # Initialize the timing list here
        self.csv_file_path = csv_file_path

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        start_time = time.time()
        flip_applied = False  # Track whether the flip was applied
        print("RandomHorizontalFlip")

        if torch.rand(1) < self.p:
            image = F.hflip(image)
            flip_applied = True
            if target is not None:
                width, _ = get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints

        # Measure the duration of the transformation
        duration = time.time() - start_time

        # Log transformation details for every image
        if target and 'image_id' in target:
            tensor_id = str(target['image_id'])
        else:
            tensor_id = "unknown"  # In case target or image_id is missing

        image_name = get_image_name(tensor_id)
        size = get_image_size(image)

        # Log details whether or not the image was flipped
        self.flip_times.append({
            "Image ID": image_name,
            "Image Size": size,
            "Flip Applied": flip_applied,  # Log whether the flip was applied
            "Time (ms)": duration * 1000
        })

        # Write to CSV
        with open(self.csv_file_path['RandomHorizontalFlip'], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([image_name, size, duration * 1000, flip_applied])

        # Print log for debugging
        print(f"rnouaj-RandomHorizontalFlip. Image: {image_name}, size: {size} - {int(size[0]) * int(size[1])}, "
              f"time: {duration * 1000:.2f} ms, flip applied: {flip_applied}")

        return image, target


class ToTensor(nn.Module):
    def __init__(self, csv_file_path):
        super().__init__()
        self.tensor_times = []  # Initialize the timing list here
        self.csv_file_path = csv_file_path

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        start_time = time.time()
        print('ToTensor')


        image = F.to_tensor(image)
        duration = time.time() - start_time
        tensor_id = str(target['image_id'])
        image_name = get_image_name(tensor_id)
        size = get_image_size(image)

        self.tensor_times.append({
            "Image ID": image_name,
            "Image Size": size,  # Image size
            "Time (ms)": duration*1000 
        })
        with open(self.csv_file_path['ToTensor'], mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([image_name, size, duration * 1000, "T"]) 


        return image, target
class RandomIoUCrop(nn.Module):
    def __init__(self, csv_file_path, min_scale: float = 0.3, max_scale: float = 1.0, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0, sampler_options: Optional[List[float]] = None, trials: int = 40):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials
        self.RandomIOU_times = []
        self.csv_file_path = csv_file_path

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        
        
        print("RandomIoUCrop")

        start_time = time.time()
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = get_image_size(image)
        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                duration = time.time() - start_time  # Time the transformation
                tensor_id = str(target.get('image_id', 'unknown'))  # Handle missing image_id
                image_name = get_image_name(tensor_id)
                size = get_image_size(image)

                # Append transformation time
                self.RandomIOU_times.append({
                    "Image ID": image_name,
                    "Image Size": size,
                    "Time (ms)": duration * 1000
                })

                # Write to CSV
                with open(self.csv_file_path['RandomIoUCrop'], mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([image_name, size, duration * 1000,'F'])

                return image, target

       

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue
                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(boxes, torch.tensor([[left, top, right, bottom]],
                                                                         dtype=boxes.dtype, device=boxes.device))
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                
        # Log transformation time
        duration = time.time() - start_time
        tensor_id = str(target.get('image_id', 'unknown'))  # Handle missing image_id
        image_name = get_image_name(tensor_id)
        size = get_image_size(image)

        # Append transformation time
        self.RandomIOU_times.append({
            "Image ID": image_name,
            "Image Size": size,
            "Time (ms)": duration * 1000
        })

        # Write to CSV
        with open(self.csv_file_path['RandomIoUCrop'], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([image_name, size, duration * 1000, "T"])

        # Print log for debugging
        print(f"rnouaj-RandomIoUCrop. Image: {image_name}, size: {size} - {int(size[0]) * int(size[1])}, "
              f"time: {duration * 1000:.2f} ms")

        return image, target


class RandomZoomOut(nn.Module):
    def __init__(self, csv_file_path, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1., 4.), p: float = 0.5):
        super().__init__()
        if fill is None:
            fill = [0., 0., 0.]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1. or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p
        self.csv_file_path = csv_file_path
        self.RandomZoomOut_times = []

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0


    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        
        
        start_time = time.time()  # Start timing the transformation
        print("RandomZoomOut")


        if torch.rand(1) < self.p:
            end_time2= time.time() - start_time
            tensor_id = str(target['image_id'])
            image_name = get_image_name(tensor_id)
            size = get_image_size(image)

            # Log transformation details for RandomZoomOut
            with open(self.csv_file_path['RandomZoomOut'], mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([image_name, size, end_time2 * 1000, "F"])  # Log in milliseconds

            return image, target

        orig_w, orig_h = get_image_size(image)

        # Transformation logic (zooming out)
        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(_is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h):, :] = \
                image[..., :, (left + orig_w):] = v


        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        # Calculate transformation time
        duration = time.time() - start_time
        tensor_id = str(target['image_id'])
        image_name = get_image_name(tensor_id)
        size = get_image_size(image)

        # Log transformation details for RandomZoomOut
        with open(self.csv_file_path['RandomZoomOut'], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([image_name, size, duration * 1000, "T"])  # Log in milliseconds

        print(f"rnouaj-RandomZoomOut. Image: {image_name}, size: {size}, time: {duration * 1000:.2f} ms")
        return image, target


class RandomPhotometricDistort(nn.Module):

    ##7 transformations are applied here
    def __init__(self, csv_file_path, contrast: Tuple[float] = (0.5, 1.5), saturation: Tuple[float] = (0.5, 1.5),
                 hue: Tuple[float] = (-0.05, 0.05), brightness: Tuple[float] = (0.875, 1.125), p: float = 0.5):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p
        self.csv_file_path = csv_file_path
        self.RandomPhotometricDistort_times=[]

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        start_time = time.time()
        print("RandomPhotometricDistort")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)
        transformations_applied = []


        if r[0] < self.p:
            image = self._brightness(image)
            transformations_applied.append('Brightness')

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)
                transformations_applied.append('contrast')

        if r[3] < self.p:
            image = self._saturation(image)
            transformations_applied.append('saturation')

        if r[4] < self.p:
            image = self._hue(image)
            transformations_applied.append('hue')

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)
                transformations_applied.append('contrast if not before')

        if r[6] < self.p:
            channels = get_image_num_channels(image)
            permutation = torch.randperm(channels)

            is_pil = _is_pil_image(image)
            if is_pil:
                image = F.to_tensor(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)
            transformations_applied.append('Channel shuffle')

        duration = time.time() - start_time
        tensor_id = str(target['image_id'])
        image_name = get_image_name(tensor_id)
        size = get_image_size(image)
        self.RandomPhotometricDistort_times.append({
            "Image ID": image_name,
            "Image Size": size,  # Image size
            "Time (ms)": duration*1000 
        })
        with open(self.csv_file_path['RandomPhotometricDistort'], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([image_name, size, duration * 1000,transformations_applied]) 

    

        print("rnouaj-RandomPhotometricDistort. Image:", image_name, "size:(", size,") -",int(size[0])*int(size[1]), "time:", duration*1000, "ms")
        return image, target



