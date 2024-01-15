from typing import Callable, List

import torch
from PIL import Image
from torchvision import transforms

from src.bigearthnet_dataset.BEN_lmdb_utils import band_combi_to_mean_std

class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


def build_transform_pipeline(dataset, aug_cfg, model_cfg) -> transforms.Compose:
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node."""

    mean, std = band_combi_to_mean_std(model_cfg.data.num_bands)

    augmentations = []

    augmentations.append(transforms.ToTensor())

    if aug_cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                aug_cfg.crop_size,
                scale=(aug_cfg.rrc.crop_min_scale, aug_cfg.rrc.crop_max_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                aug_cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if aug_cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=aug_cfg.horizontal_flip.prob))

    if aug_cfg.vertical_flip.prob:
        augmentations.append(transforms.RandomVerticalFlip(p=aug_cfg.vertical_flip.prob))

    augmentations.append(transforms.Normalize(mean=mean, std=std))

    augmentations = transforms.Compose(augmentations)
    return augmentations


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)

