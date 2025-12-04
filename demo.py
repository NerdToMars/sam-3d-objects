import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")

for i in range(26):
    mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=i)
    output = inference(image, mask, seed=42)
    output["gs"].save_ply(f"splat_{i}.ply")
    print(f"Your reconstruction has been saved to splat_{i}.ply")
