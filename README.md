# k-Space

### Problems

import error : umap (in jupyter notebook)

import error : faiss (in jupyter notebook)

## Models

- ResNet50
    - pretrained-IN:
    - pretrained-SIN:
- Vision Transformer
    - pretrained-IN:

### Dataset(ImageNet)

- Original
- Stylized
- Cropped
- Rotated
- Gaussian Noise
- Adversarial Attack
    - PGD
    - DeepFool
    - OnePixel

### Metrics

- mean
- variance
- std
- median
- max
- min
- kNN accuracy

## Source Files

- Getting SoftMax dict → ~~/model-vs-human/{something}.py~~
    - to /softmax_dict/
- Getting Neighbors and k-Predictions dict → /ForShare/get_neighbors.py
    - to /neighbors_dict/
    - to /prediction_dict/
- Make Metrics Visalization → /ForShare/k-Space_Metrics.ipynb
- Make Space Visalization → /ForShare/k-Space_Vis.ipynb
- For Space Explanation → /ForShare/Vis2D_sample.ipynb
- For Metrics Explanation → /ForShare/kSpace_sample.ipynb

### Contents of Dict

- keys = ImageNet classes
    - ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
- values = (the # of one class datas, the # of k range=1280)
