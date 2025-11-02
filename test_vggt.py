import torch
import sys
sys.path.append("llava/model/vggt")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
#image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
#images = load_and_preprocess_images(image_names).to(device)
image = Image.open("image.jpeg")
print(np.array(image).shape)
image = np.array(image.resize((336,336)))
image = torch.tensor(image).permute(2,0,1)
images = [image / 255]
images = torch.stack(images).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]
        print(images.shape)  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
    print(aggregated_tokens_list[0].shape)
    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

print(depth_map)
depth_map = depth_map.cpu().detach().numpy()
depth_map = np.squeeze(depth_map[0][0],axis=-1)
depth_map = (depth_map*255).astype('uint8')
plt.imsave("depth1_viridis.png", depth_map, cmap="viridis", vmin=0.0, vmax=depth_map.max())
