# SAM 3D

SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction.  If you’re looking for SAM 3D Body, [click here](https://github.com/facebookresearch/sam-3d-body).

# SAM 3D Objects

**SAM 3D Team**, [Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra)\*, [Fu-Jen Chu](https://fujenchu.github.io/)\*, [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao)\*, [Kevin J Liang](https://kevinjliang.github.io/)\*, [Alexander Sax](https://alexsax.github.io/)\*, [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra)\*, [Weiyao Wang](https://sites.google.com/view/weiyaowang/home)\*, [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/)⚬, [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/)⚬, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra)⚬, [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/)⚬, [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)⚬, [Piotr Dollár](https://pdollar.github.io/)†, [Georgia Gkioxari](https://georgiagkioxari.com/)†, [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†§, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†§

***Meta Superintelligence Labs***

*Core contributor (Alphabetical, Equal Contribution), ⚬Intern, †Project leads, §Equal Contribution

[[`Paper`](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)] [[`Code`](https://github.com/facebookresearch/sam-3d-objects)] [[`Website`](https://ai.meta.com/sam3d/)] [[`Demo`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)] [[`Blog`](https://ai.meta.com/blog/sam-3d/)] [[`BibTeX`](#citing-sam-3d-objects)] [[`Roboflow`](https://blog.roboflow.com/sam-3d/)]

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.


<p align="center"><img src="doc/intro.png"/></p>

-----

<p align="center"><img src="doc/arch.png"/></p>

## Latest updates

**11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

## Single or Multi-Object 3D Generation

SAM 3D Objects can convert masked objects in an image, into 3D models with pose, shape, texture, and layout. SAM 3D is designed to be robust in challenging natural images, handling small objects and occlusions, unusual poses, and difficult situations encountered in uncurated natural scenes like this kidsroom:

<p align="center">
  <img src="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" width="55%"/>
  <img src="doc/kidsroom_transparent.gif" width="40%"/>
</p>

For a quick start, run `python demo.py` or use the the following lines of code:

```python
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

For  more details and multi-object reconstruction, please take a look at out two jupyter notebooks:
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)


## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Objects model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Objects project was made possible with the help of many contributors.

Robbie Adkins,
Paris Baptiste,
Karen Bergan,
Kai Brown,
Michelle Chan,
Ida Cheng,
Khadijat Durojaiye,
Patrick Edwards,
Daniella Factor,
Facundo Figueroa,
Rene  de la Fuente,
Eva Galper,
Cem Gokmen,
Alex He,
Enmanuel Hernandez,
Dex Honsa,
Leonna Jones,
Arpit Kalla,
Kris Kitani,
Helen Klein,
Kei Koyama,
Robert Kuo,
Vivian Lee,
Alex Lende,
Jonny Li,
Kehan Lyu,
Faye Ma,
Mallika Malhotra,
Sasha Mitts,
William Ngan,
George Orlin,
Peter Park,
Don Pinkus,
Roman Radle,
Nikhila Ravi,
Azita Shokrpour,
Jasmine Shone,
Zayida Suber,
Phillip Thomas,
Tatum Turner,
Joseph Walker,
Meng Wang,
Claudette Ward,
Andrew Westbury,
Lea Wilken,
Nan Yang,
Yael Yungster


```
Complete Processing Flow
Step 1: Input Dict
ss_input_dict = {
    "image": [B, 3, 518, 518],           # Cropped RGB
    "rgb_image": [B, 3, 518, 518],       # Full RGB
    "mask": [B, 3, 518, 518],            # Cropped mask
    "rgb_image_mask": [B, 3, 518, 518],  # Full RGB+mask
    "pointmap": [B, 3, 256, 256],        # Cropped 3D points
    "rgb_pointmap": [B, 3, 256, 256],    # Full 3D points
}
Step 2: Embedder Processing
# Embedder 1: DINOv2 for images
tokens_1a = dino1(image) + pos_emb["cropped"]          # [B, 1370, 1024]
tokens_1b = dino1(rgb_image) + pos_emb["full"]         # [B, 1370, 1024]

# Embedder 2: DINOv2 for masks
tokens_2a = dino2(mask) + pos_emb["cropped"]           # [B, 1370, 1024]
tokens_2b = dino2(rgb_image_mask) + pos_emb["full"]    # [B, 1370, 1024]

# Embedder 3: PointPatchEmbed for depth
tokens_3a_raw = pointembed(pointmap)                    # [B, 1024, 512]
tokens_3a = projection(tokens_3a_raw) + pos_emb["cropped"]  # [B, 1024, 1024]

tokens_3b_raw = pointembed(rgb_pointmap)                # [B, 1024, 512]
tokens_3b = projection(tokens_3b_raw) + pos_emb["full"]     # [B, 1024, 1024]
Step 3: Concatenation
final_tokens = concat([
    tokens_1a,  # Image cropped
    tokens_1b,  # Image full
    tokens_2a,  # Mask cropped
    tokens_2b,  # Mask full
    tokens_3a,  # Pointmap cropped
    tokens_3b,  # Pointmap full
], dim=1)


DINOv2 outputs 1370 tokens (with registers excluded)
1370 × 4 (two DINOv2 embedders, each processing 2 inputs) = 5480
PointEmbed outputs 1024 tokens × 2 inputs = 2048
Total: 5480 + 2048 = 7528 tokens

🎯 Positional Encoding Groups
The Mapping:
self.positional_embed_map = {
    "cropped": 0,  # Index 0
    "full": 1,     # Index 1
}
Learned Embeddings:
self.idx_emb = nn.Parameter(torch.empty(2, 1024))
# Shape: [2, 1024]
# idx_emb[0] = embedding for "cropped" inputs
# idx_emb[1] = embedding for "full" inputs
Why Needed?
Problem: All 6 inputs go to the same cross-attention. How does the model distinguish them? Solution: Add learnable positional encodings based on input type:
"cropped" inputs (3 total):
  - image (cropped)       + idx_emb[0]
  - mask (cropped)        + idx_emb[0]
  - pointmap (cropped)    + idx_emb[0]

"full" inputs (3 total):
  - rgb_image (full)      + idx_emb[1]
  - rgb_image_mask (full) + idx_emb[1]
  - rgb_pointmap (full)   + idx_emb[1]
The model learns:
idx_emb[0] → "this is object-centric/zoomed-in information"
idx_emb[1] → "this is scene-wide/contextual information"
📊 Visual Summary
┌────────────────────────────────────────────────────────────┐
│                    INPUT RGBA IMAGE                         │
│                    [B, 4, H, W]                            │
└────────────────────────────────────────────────────────────┘
                            ↓
            ┌───────────────┴───────────────┐
            ↓                               ↓
    ┌──────────────┐               ┌──────────────┐
    │   CROPPED    │               │     FULL     │
    │   (object)   │               │   (context)  │
    └──────────────┘               └──────────────┘
            ↓                               ↓
    ┌───────┴───────┐           ┌───────────┴──────────┐
    ↓       ↓       ↓           ↓           ↓          ↓
  RGB    Mask  PointMap      RGB        RGB+Mask   PointMap
  518×518 518×518 256×256    518×518     518×518    256×256
    ↓       ↓       ↓           ↓           ↓          ↓
┌─────────────────────────────────────────────────────────────┐
│                   EMBEDDERS                                  │
├─────────────────────────────────────────────────────────────┤
│  DINOv2_1      DINOv2_2    PointEmbed                       │
│  (images)      (masks)     (depth)                          │
│     ↓              ↓            ↓                            │
│  1370 tok      1370 tok     1024 tok                        │
│  ×2 inputs     ×2 inputs    ×2 inputs                       │
│  = 2740        = 2740       = 2048                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
            Add positional encoding per group
            ("cropped" or "full")
                            ↓
            ┌───────────────┴────────────────┐
            ↓                                ↓
    Cropped tokens (3)              Full tokens (3)
    + pos_emb[0]                    + pos_emb[1]
                            ↓
            ┌───────────────┴────────────────┐
            │      Project all to 1024-dim    │
            │      (PointEmbed: 512→1024)     │
            └─────────────────────────────────┘
                            ↓
            ┌───────────────┴────────────────┐
            │   Concatenate along token dim   │
            │   [B, 7528, 1024]               │
            └─────────────────────────────────┘
                            ↓
            ┌───────────────┴────────────────┐
            │   Cross-Attention in Generator  │
            │   Q: Shape/Pose latents        │
            │   K,V: These 7528 tokens       │
            └─────────────────────────────────┘
💡 Key Design Insights
1. Multi-Scale Context
Cropped: High-res object details
Full: Scene layout and context
Both needed for accurate 3D reconstruction
2. Multi-Modal Features
RGB: Appearance, texture, semantics
Mask: Shape, boundaries, silhouette
Depth: 3D geometry, metric scale
3. Separate Encoders
Each modality gets its own encoder
Prevents feature entanglement
Allows specialized processing
4. Positional Encoding Groups
Distinguishes cropped vs. full inputs
Learned (not fixed like sinusoidal)
Shared across modalities (all cropped share same encoding)
5. Resolution Trade-offs
DINOv2: 518×518 (high-res for appearance)
PointMap: 256×256 (sufficient for geometry, faster)
6. Token Counts
DINOv2: 1370 tokens/input (37×37 spatial grid)
PointMap: 1024 tokens/input (32×32 spatial grid)
Total: 7528 condition tokens
```

## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

```
@article{sam3dteam2025sam3d3dfyimages,
      title={SAM 3D: 3Dfy Anything in Images}, 
      author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
      year={2025},
      eprint={2511.16624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16624}, 
}
```
