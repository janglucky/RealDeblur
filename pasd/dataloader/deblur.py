import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(root: str) -> List[Path]:
    root_path = Path(root)
    return sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def _build_stem_index(paths: Iterable[Path]) -> Dict[str, Path]:
    index = {}
    for path in paths:
        index.setdefault(path.stem, path)
    return index


class DeblurPairDataset(data.Dataset):
    def __init__(
        self,
        blur_dir: str,
        sharp_dir: str,
        image_size: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
    ):
        super().__init__()
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.pairs = self._collect_pairs()
        if not self.pairs:
            raise ValueError(
                f"No blur/sharp pairs found. blur_dir={self.blur_dir}, sharp_dir={self.sharp_dir}"
            )

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        blur_paths = _list_images(str(self.blur_dir))
        sharp_paths = _list_images(str(self.sharp_dir))
        blur_by_stem = _build_stem_index(blur_paths)
        pairs = []

        for sharp_path in sharp_paths:
            rel_path = sharp_path.relative_to(self.sharp_dir)
            blur_path = self.blur_dir / rel_path
            if not blur_path.is_file():
                blur_path = blur_by_stem.get(sharp_path.stem)
            if blur_path is not None and blur_path.is_file():
                pairs.append((blur_path, sharp_path))

        return pairs

    def _resize_if_needed(self, blur: Image.Image, sharp: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if blur.size != sharp.size:
            blur = blur.resize(sharp.size, Image.BICUBIC)

        if self.image_size <= 0:
            return blur, sharp

        width, height = sharp.size
        min_side = min(width, height)
        if min_side >= self.image_size:
            return blur, sharp

        scale = self.image_size / min_side
        new_size = (round(width * scale), round(height * scale))
        return blur.resize(new_size, Image.BICUBIC), sharp.resize(new_size, Image.BICUBIC)

    def _crop(self, blur: Image.Image, sharp: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.image_size <= 0:
            width, height = sharp.size
            width = width // 8 * 8
            height = height // 8 * 8
            return TF.crop(blur, 0, 0, height, width), TF.crop(sharp, 0, 0, height, width)

        width, height = sharp.size
        if self.center_crop:
            top = max((height - self.image_size) // 2, 0)
            left = max((width - self.image_size) // 2, 0)
        else:
            top = random.randint(0, height - self.image_size)
            left = random.randint(0, width - self.image_size)

        blur = TF.crop(blur, top, left, self.image_size, self.image_size)
        sharp = TF.crop(sharp, top, left, self.image_size, self.image_size)
        return blur, sharp

    def __getitem__(self, index: int):
        blur_path, sharp_path = self.pairs[index]
        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")

        blur, sharp = self._resize_if_needed(blur, sharp)
        blur, sharp = self._crop(blur, sharp)

        if self.random_flip and random.random() < 0.5:
            blur = TF.hflip(blur)
            sharp = TF.hflip(sharp)

        sharp_tensor = TF.to_tensor(sharp) * 2.0 - 1.0
        blur_tensor = TF.to_tensor(blur)

        return {
            "pixel_values": sharp_tensor,
            "conditioning_pixel_values": blur_tensor,
            "blur_path": os.fspath(blur_path),
            "sharp_path": os.fspath(sharp_path),
        }

    def __len__(self):
        return len(self.pairs)
