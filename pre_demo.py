import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from statistics import mean
from torch.nn.functional import threshold, normalize
from pre_dataset import dataset
import torch
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
# orig是最初的sam
model_type = 'vit_b'
checkpoint = './model/vit_b.pth'
device = 'cuda:7'
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device)
sam_model = sam_model_registry[model_type](checkpoint='./model/12.pth')
sam_model.to(device)
predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)


path=r"Z:\lulc_dataset\Globe230k\patch_image\mwqh4t87d6nn_lines_2_samples_3.jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array([175,135,270,260])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=None,
    multimask_output=False,
)


_, axs = plt.subplots(1, 2, figsize=(25, 25))


axs[0].imshow(image)
show_mask(masks_orig, axs[0])
show_box(list(input_bbox), plt.gca())
axs[0].set_title('Mask with no prompt', fontsize=26)
axs[0].axis('off')


axs[1].imshow(image)
show_mask(masks_tuned, axs[1])
axs[1].set_title('Mask with one bbox prompt', fontsize=26)
axs[1].axis('off')

plt.show()









