# GraPL
Segmentation via **Gra**ph Cuts at the **P**atch **L**evel

GraPL is implemented inside the `GraPL` module directory. Our evaluation pipeline is also implemented in this under `evaluate.py`.

## Usage
1. Start by installing the necessary dependencies using the provided conda environment file `env.yml`.
2. Unzip `datasets/BSDS500.zip`. This file contains all of the necessary test images and their ground truths.
3. Use the included notebooks to segment images. You can use `GraPL_testbench.ipynb` to test the GraPL codebase one segmentation at a time and get a sense of typical segmentation performance. Alternatively, our major experiments can be reproduced using the various `experiment_*.ipynb` notebooks. BSDS500 is included in this codebase for easy reproducibility.