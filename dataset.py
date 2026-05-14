import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch, load_resize_img, load_resize_mask
import torch.nn.functional as F
import random
import cv2

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def _resolve_shadow_dirs(rgb_dir, val=False, plus=False):
    if os.path.isdir(os.path.join(rgb_dir, 'train_A')):
        gt_dir = 'test_C_fixed_official' if val and plus else 'train_C'
        return gt_dir, 'train_A', 'train_B'
    if os.path.isdir(os.path.join(rgb_dir, 'shadow_imgs')):
        return 'shadowfree_imgs', 'shadow_imgs', 'shadow_masks'
    raise FileNotFoundError(
        "Expected either train_A/train_B/train_C or "
        "shadow_imgs/shadow_masks/shadowfree_imgs under {}".format(rgb_dir)
    )

def _pad_to_patch(clean, noisy, mask, patch_size):
    _, h, w = clean.shape
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return clean, noisy, mask

    pad = (0, pad_w, 0, pad_h)
    clean = F.pad(clean, pad, mode='reflect')
    noisy = F.pad(noisy, pad, mode='reflect')
    mask = F.pad(mask.unsqueeze(0), pad, mode='reflect').squeeze(0)
    return clean, noisy, mask

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None, plus=False):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        gt_dir, input_dir, mask_dir = _resolve_shadow_dirs(rgb_dir)
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        clean, noisy, mask = _pad_to_patch(clean, noisy, mask, ps)
        H = clean.shape[1]
        W = clean.shape[2]
        r = 0 if H == ps else np.random.randint(0, H - ps + 1)
        c = 0 if W == ps else np.random.randint(0, W - ps + 1)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        mask = mask[r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None, plus=False):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        gt_dir, input_dir, mask_dir = _resolve_shadow_dirs(rgb_dir, val=True, plus=plus)
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # clean = torch.from_numpy(np.float32(load_resize_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_resize_img(self.noisy_filenames[tar_index])))
        # mask = load_resize_mask(self.mask_filenames[tar_index])

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename


##################################################################################################
class DataLoaderSBUVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderSBUVal, self).__init__()

        self.target_transform = target_transform

        # gt_dir = 'test_C_fixed_official'

        input_dir = 'frames'
        mask_dir = 'test_B'

        mask_file = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        clean_files = []
        input_files = []
        mask_files = []
        for file in mask_file:
            for m in os.listdir(os.path.join(rgb_dir, input_dir, file[:-9])):
                input_files.append(os.path.join(rgb_dir, input_dir, file[:-9], m))
                clean_files.append(os.path.join(rgb_dir, input_dir, file[:-9], m))
                mask_files.append(os.path.join(rgb_dir, mask_dir,  file))

        self.clean_filenames = [x for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [x for x in input_files if is_png_file(x)]
        self.mask_filenames = [x for x in mask_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        # clean = torch.from_numpy(np.float32(load_resize_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_resize_img(self.noisy_filenames[tar_index])))
        # mask = load_resize_mask(self.mask_filenames[tar_index])

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], size=(640, 480))
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename, mask_filename
