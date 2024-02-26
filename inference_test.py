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
from tqdm import tqdm
from os.path import join
from time import perf_counter as time
from monai.transforms import Compose, EnsureChannelFirstd 
from monai.inferers import SliceInferer


DATA_PATH='/dtu/3d-imaging-center/courses/02510/data/CovidHeart/covid_small'
image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()
train_label = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'mask_0.npy')))
val_label = torch.from_numpy(np.load(join(DATA_PATH, 'val', 'mask_0.npy')))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.2,  # Read about dropout here: https://www.deeplearningbook.org/contents/regularization.html#pf20
)

checkpoint = torch.load('models/model_2d_checkpoint.pth')
model.load_state_dict(checkpoint['model'])


num =image.shape[0]//8
for idx in range(3):  
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,

    )
    val_transforms = Compose([
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
    ])
    #train_data = DataClass(split='train', transform=train_transforms, root=MEDMNIST_ROOT)
    val_dataset = CacheDataset(
        data=[{'image':image[:,:,num*1:num*(1+1)]}],
        #data=image,
        transform=val_transforms,
        num_workers=8,
        cache_rate=1.0
    )
    VAL_BATCH_SIZE = 1
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Just use the main thread for now, we just need it for visualization
    )
    batch = next(iter(val_loader)) 
    from monai.inferers import SliceInferer
    model.cuda()
    model.eval()
    with torch.no_grad():
        image_b = batch['image'].cuda()
        inferer = SliceInferer(roi_size=(96, 96),spatial_dim=1,progress=True)
        output = inferer(image_b, model).softmax(dim=1).cpu().numpy()
    #output_test = output.softmax(dim=1).cpu().numpy()
    output_test = np.uint8(output[0, 0] * 255)

    from skimage.io import imsave
    import nibabel as nib

    np.save("models/index_"+ str(idx) +".npy", output_test)  # For TomViz
    print('We finished',idx)