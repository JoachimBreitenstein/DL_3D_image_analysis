# Import libraries 
from os.path import join
from time import perf_counter as time

import matplotlib.pyplot as plt
import monai
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
import numpy as np
from skimage.measure import label as skimage_label, regionprops
import torch
from monai.metrics import compute_dice
import numpy

## Load data
DATA_PATH='/dtu/3d-imaging-center/courses/02510/data/CovidHeart/covid_small'
image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()
train_label = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'mask_0.npy')))
val_label = torch.from_numpy(np.load(join(DATA_PATH, 'val', 'mask_0.npy')))

#hyperparameters
PATCH_SIZE=(96,) * 3         # Size of crops
PROB_FOREGROUND_CENTER=0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)
NUM_EPOCHS = 1
LR = 1e-4
TRAIN_BATCH_SIZE=16
BATCHES_PER_EPOCHS=150
VAL_BATCH_SIZE=16

from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    FgBgToIndicesd,
    LabelToMaskd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandAxisFlipd,
)

train_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    CopyItemsd(keys=['label'], times=1, names=['mask']),                                                  # Copy label to new image mask
    LabelToMaskd(keys=['mask'], select_labels=[1, 2], merge_channels=True),                               # Convert mask to binary mask showing where labels are
    FgBgToIndicesd(keys=['mask'], fg_postfix='_fg_indices', bg_postfix='_bg_indices'),                    # Precompute indices of labelled voxels
    RandCropByPosNegLabeld(keys=['image', 'label', 'mask'], label_key='label', spatial_size=PATCH_SIZE,   # Extract random crop
                            pos=PROB_FOREGROUND_CENTER, neg=(1.0 - PROB_FOREGROUND_CENTER),
                            num_samples=1, fg_indices_key='mask_fg_indices', bg_indices_key='mask_bg_indices'),
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 1)),                        # Randomly rotate
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(1, 2)),                        # Randomly rotate
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 2)),                        # Randomly rotate
    RandAxisFlipd(keys=['image', 'label', 'mask'], prob=0.1),                                             # Randomly flip
])

class RepeatedCacheDataset(monai.data.CacheDataset):
    """
    CacheDataset that repeats the data.
    """

    def __init__(self, *args, num_repeats: int = 1, **kwargs):
        """
        Args:
            *args: Arguments to pass to CacheDataset.
            num_repeats (int): Number of times to repeat the data.
            **kwargs: Keyword arguments to pass to CacheDataset.
        """
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats


    def __len__(self):
        """Returns number of items in the dataset."""
        return super().__len__() * self.num_repeats


    def __getitem__(self, index):
        """Returns the item at the given index."""
        index = index % super().__len__()
        return super().__getitem__(index)

## Use RepeatedCacheDataset
train_dataset = RepeatedCacheDataset(
    data=[{ 'image': image, 'label': train_label }],
    num_repeats=BATCHES_PER_EPOCHS * TRAIN_BATCH_SIZE,
    transform=train_transforms,
    num_workers=8,
    cache_rate=1.0,
    copy_cache=False  # Important to avoid slowdowns
)    

from typing import Tuple, List, Dict

def extract_label_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> List[Dict[str, np.ndarray]]:
    """
    Extract patches from image where label is non-zero.

    For each connected component in label, extract the bounding box.
    Split the bounding box into overlapping patches of size patch_size.
    Extract the patches from image and label.
    Return the patches as a list of { 'image': ..., 'label': ... } dicts.

    Args:
        image (np.ndarray): Image to extract patches from.
        label (np.ndarray): Label to extract patches from.
        patch_size (tuple): Size of the patches to extract.

    Returns:
        list: List of patches as { 'image': ..., 'label': ..., 'mask': ... } dicts.
    """
    props = regionprops(skimage_label(label > 0))  # Extract connected components of labeled voxels
    patches = []
    for pp in props:
        # Extract bounding box for connected component
        cc_min = pp.bbox[:3]
        cc_max = pp.bbox[3:]

        # Extract patches covering the bounding box
        for z in range(cc_min[0] - patch_size[0] // 2, cc_max[0] + patch_size[0] // 2, patch_size[0] // 2):
            for y in range(cc_min[1] - patch_size[1] // 2, cc_max[1] + patch_size[1] // 2, patch_size[1] // 2):
                for x in range(cc_min[2] - patch_size[2] // 2, cc_max[2] + patch_size[2] // 2, patch_size[2] // 2):
                    # Ensure patch is within image bounds
                    z_begin = max(z, 0)
                    y_begin = max(y, 0)
                    x_begin = max(x, 0)
                    z_end = min(z + patch_size[0], image.shape[0])
                    y_end = min(y + patch_size[1], image.shape[1])
                    x_end = min(x + patch_size[2], image.shape[2])

                    patch_label = label[z_begin:z_end, y_begin:y_end, x_begin:x_end]
                    if not patch_label.any():
                        # Skip empty patches
                        continue
                    patch_image = image[z_begin:z_end, y_begin:y_end, x_begin:x_end]

                    if patch_image.shape != patch_size:
                        # Pad patch if it is smaller than patch_size
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_image.shape[d])
                        patch_image = np.pad(patch_image, pad_size, 'constant', constant_values=0)
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_label.shape[d])
                        patch_label = np.pad(patch_label, pad_size, 'constant', constant_values=0)

                    patches.append({ 'image': patch_image, 'label': patch_label, 'mask': patch_label > 0 })

    return patches



val_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label', 'mask'], channel_dim='no_channel'),
])
val_dataset = CacheDataset(
    data=extract_label_patches(image, val_label, PATCH_SIZE),
    transform=val_transforms,
    num_workers=8,
    cache_rate=1.0
)

## Define model 
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    act='ReLU',
    dropout=0.2,  # Read about dropout here: https://www.deeplearningbook.org/contents/regularization.html#pf20
)


loss_fn = monai.losses.MaskedDiceLoss(include_background=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()


## Dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,  # Don't shuffle since we use random crops
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

model.cuda()


## Training loop 
all_train_losses = []
all_val_losses = []
train_times = []
val_times = []
all_dice_metrics = []
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    mean_train_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.train()
    for batch in train_loader:
        image_b = batch['image'].as_tensor().cuda(non_blocking=True)
        label = batch['label'].as_tensor().cuda(non_blocking=True)
        mask = batch['mask'].as_tensor().cuda(non_blocking=True)
        label = one_hot(label, num_classes=3)
        label = label[:, 1:]

        ### YOUR CODE HERE ###
        with torch.cuda.amp.autocast():
            pred = model(image_b)
            loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=None)

        mean_train_loss += loss.detach() * len(image_b)
        num_samples += len(image_b)
        step += 1

    train_time = time() - t0
    train_times.append(train_time)
    mean_train_loss = mean_train_loss / num_samples
    all_train_losses.append(mean_train_loss.item())

    mean_val_loss = 0
    num_samples = 0
    step = 0
    dice_sum = 0
    t0 = time()
    model.eval()
    for batch in val_loader:
        image_b = batch['image'].as_tensor().cuda(non_blocking=True)
        label = batch['label'].as_tensor().cuda(non_blocking=True)
        mask = batch['mask'].as_tensor().cuda(non_blocking=True)

        with torch.no_grad():
            label = one_hot(label, num_classes=3)
            label = label[:, 1:]

            ### YOUR CODE HERE ###
            with torch.cuda.amp.autocast():
                pred = model(image_b)
                loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)
                
        dice = 1-loss.item()
        mean_val_loss += loss * len(image_b)
        num_samples += len(image_b)
        step += 1
        
    dice_sum += dice
    all_dice_metrics.append(dice)
    mean_val_loss = mean_val_loss / num_samples
    
    val_time = time() - t0
    val_times.append(val_time)
    all_val_losses.append(mean_val_loss.item())
        
    if mean_val_loss.item() < best_val_loss:
        best_val_loss = mean_val_loss
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'train_losses': all_train_losses,
            'val_losses': all_val_losses,
            'Dice score': all_dice_metrics,
        }, 'models_200/model_checkpoint_3D.pth')

with open('models_200/03_05_3D_baseline.txt', 'w') as f:
    f.write('train times: ' + str(train_times) + ' seconds\n')
    f.write('eval times: ' + str(val_times) + ' seconds\n')
    f.write('all train losses: ' + str(all_train_losses) + '\n')
    f.write('all val losses: ' + str(all_val_losses) + '\n')
    f.write('mean train time: ' + str(np.mean(train_times[1:])) + '\n')
    f.write('mean eval time: ' + str(np.mean(val_times)) + '\n')
    f.write('Dice score: ' + str(all_dice_metrics) + '\n')