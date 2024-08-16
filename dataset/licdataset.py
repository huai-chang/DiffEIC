from typing import Sequence, Dict, Union
import time

import numpy as np
from PIL import Image
import torch.utils.data as data

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr

class LICDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
    ) -> "LICDataset":
        super(LICDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.use_rot = use_rot

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            # assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        # hwc, [0, 255] to [0, 1], float32
        img_gt = (pil_img_gt / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=self.use_rot, return_status=False)
        
        # [-1, 1]
        target = (img_gt * 2 - 1).astype(np.float32)
        # [0, 1]
        source = img_gt.astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source)

    def __len__(self) -> int:
        return len(self.paths)
