# 🌿 Plant Disease Detection

## Dataset

[Plant Village Dataset from kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Classes used

- Tomato__Early_blight
- Tomato__healthy
- Tomato__Late_blight
- Tomato__Target_Spot
- Tomato__Tomato_mosaic_virus


##  Project Workflow



### 1. Dataset Preparation & Splitting

- Collected dataset of leaf images (color version).

- Organized into train/validation/test directories using subsplit.py.

- Used `tf.keras.utils.image_dataset_from_directory` for loading datasets.

- Applied:

- Resizing: 128x128 (for baseline CNN) and 244x244 (for MobileNetV2).
- Normalization: pixel values scaled to `[0,1]`.
- Shuffling and batching.



Output:

- `train_ds`: training set.
- `val_ds`: validation set.
- `test_ds`: final evaluation set.



---



### 2. Baseline CNN

- Trained a simple Convolutional Neural Network from scratch to establish a reference accuracy.



Architecture:

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling

- Flatten

- Dense (hidden) → ReLU
- Dense (output) → Softmax



Training:

- Optimizer: Adam

- Loss: Categorical Crossentropy

- EarlyStopping + ModelCheckpoint callbacks



Purpose:

-  baseline model to compare more advanced architectures(i.e. MobileNetV2).
- extract information about dataset quality and class separability.



---



### 3. MobileNet (Pretrained CNN)

- Switched to MobileNet (V1), pretrained on ImageNet.

- Reason: MobileNet is lightweight and efficient, making it suitable for deployment on datasets with smaller samples.



#### Steps:

1. Imported MobileNet with `weights="imagenet"` and `include_top=False`.
2. Added custom classification head:
- Global Average Pooling
- Dense layer (Softmax, matching number of classes)



#### Advantages over Baseline CNN:

- Much fewer parameters due to depthwise separable convolutions.
- Faster training(took longer when changed to image size of 244).
- Better generalization from pretrained ImageNet features.



---



### 4. Fine-Tuning MobileNet

- After initial training with frozen MobileNet layers (feature extraction mode), performed fine-tuning.



Process:

1. Unfroze top layers of MobileNet.
2. Used a low learning rate (1e-5) to avoid forgetting.
3. Re-trained end-to-end.


Result:

- Improved accuracy compared to frozen MobileNet.
- Balanced training speed and performance.


## Setup

```bash

python3.11 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

```
