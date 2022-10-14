# k-Space

## Oct 13
Ueda has added a get_dicts folder.
This folder contains the code to output SoftMax and neighbor_dict.

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

---

## Tables and Figures to be Created.

### Figures

- Illustration Figure
- Explanation Figure
- One Figure with different k-Spaces and other graphs for different classes for ResNet. 

These all figures will be one-row figures with number of columns equal to the number of classes.
- One Figure with different k-Spaces for different classes for ViT
- One Figure with different k-Spaces for different training-samples for ResNet
- One Figure with different k-Spaces for different test-samples (One of Cropped, Rotated, Stylised or any other) for ResNet
- One Figure with different k-Spaces for different adversarial-samples (One of AutoPGD, Deepfool) for ResNet
- One Figure with different k-Spaces for different layer of ResNet for classification (possibly the last convolution layer) for ResNet
- One Figure with different k-Spaces for different layer of ResNet for other task (possibly the last convolution layer, and object detection task) for ResNet

### Table

Table 1, encompasses, Figures 3 and 4 -> Objective Comparison between Models

| Model     | Class 1 | Class 2 | Class 3 | Overall |
|-----------|---------|---------|---------|---------|
| ResNet-50 |         |         |         |         |
| ViT       |         |         |         |         |
| ...       |         |         |         |         |

Table 2, encompassing Figure 5 -> Objective Comparison between Models trained with different training analysing together with Table 1 

| Models              | Class 1 | Class 2 | Class 3 | Overall |
|---------------------|---------|---------|---------|---------|
| ResNet-50- Stylised |         |         |         |         |
| ViT-Stylised        |         |         |         |         |
| ...                 |         |         |         |         |


Table 3, encompassing Figure 6 and 7 -> Objective Comparison between Testing Samples 

| Testing Samples | Class 1 | Class 2 | Class 3 | Overall |
|-----------------|---------|---------|---------|---------|
| Stylised        |         |         |         |         |
| Cropped         |         |         |         |         |
| ...             |         |         |         |         |

Table 4, encompassing Figure 8 -> Objective Comparison between Layers


| Layers           | Class 1 | Class 2 | Class 3 | Overall |
|----------------|---------|---------|---------|---------|
| Block 4 Conv 3 |         |         |         |         |
| Block 3 Conv 3 |         |         |         |         |
| ...            |         |         |         |         |

Table 5, encompassing Figure 8 and 9 -> Objective Comparison between Tasks

| Task                | Class 1 | Class 2 | Class 3 | Overall |
|---------------------|---------|---------|---------|---------|
| Classification      |         |         |         |         |
| Objection Detection |         |         |         |         |
| ...                 |         |         |         |         |

## Experiment Settings

### Models Required

- ResNet Trained on ImageNet 1k - ResNet-Cls-1k-Normal
- ResNet Trained on Stylised ImageNet 1k - ResNet-Cls-1k-Stylised
- ViT Trained on ImageNet 1k (Careful since, the best model is trained on JFT-300M and not ImageNet 1k) ViT-Cls-1k-Normal
- ResNet Trained for Object Detection - ResNet-OD-Cls-??-Normal

Oct. 14
- ResNet-50 ImageNet1k Normal & ResNet-50 Stylized ImageNet
    
    [https://github.com/bethgelab/model-vs-human](https://github.com/bethgelab/model-vs-human)
    
- ViT-B16 ImageNet1k Normal
    
    [https://github.com/lukemelas/PyTorch-Pretrained-ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT)

### Experiments

- ResNet-Cls-1k-Normal on Normal Samples
- ResNet-Cls-1k-Normal on Different Testing Samples 
- ResNet-Cls-1k-Normal on Different layers but normal samples
- ResNet-Cls-1k-Stylised on Normal Samples
- ViT-Cls-1k-Normal on Normal Samples
- ViT-Cls-1k-Stylised on Normal Samples
- ResNet-OD-Cls-??-Normal on Normal Samples