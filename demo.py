import sys
import os

import uuid
import imageio
import numpy as np
from IPython.display import Image as ImageDisplay

# import inference code
sys.path.append("notebook")
# from inference import Inference, load_image, load_single_mask
from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

IMAGE_PATH = f"notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
# display_image(image, masks)

outputs = [inference(image, mask, seed=42) for mask in masks]

scene_gs = make_scene(*outputs)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"notebook/gaussians/multi/{IMAGE_NAME}.ply")

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"notebook/gaussians/multi/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

# notebook display
# ImageDisplay(url=f"gaussians/multi/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}",)

# load image (RGBA only, mask is embedded in the alpha channel)
# image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")

# for i in range(26):
#     mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=i)
#     output = inference(image, mask, seed=42)
#     output["gs"].save_ply(f"splat_{i}.ply")
#     print(f"Your reconstruction has been saved to splat_{i}.ply")
