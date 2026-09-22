# Adversarial Attack Detection in Chest X-Ray Images Using LPAM

## Overview

This repository contains the implementation associated with a study on the analysis and detection of adversarial perturbations in chest X-ray images using deep learning and Layer-wise Progressive Activation Mapping (LPAM).

The implementation uses a pretrained ResNet-50 model together with Grad-CAM to examine activation patterns at multiple convolutional layers. The resulting layer-wise activation maps are combined into an LPAM representation, from which activation-based metrics are calculated.

The notebook evaluates:
- Clean chest X-ray images
- FGSM (Fast Gradient Sign Method) adversarial images
- PGD (Projected Gradient Descent) adversarial images
- DeepFool adversarial images

For each image, activation-based metrics are calculated and compared with an experimentally defined threshold to classify the image as Clean or Adversarial Attack Detected.

## Dataset Information

### COVID-Pneumonia-Normal Chest X-Ray Dataset

The implementation uses the **COVID-Pneumonia-Normal Chest X-Ray Images** dataset.

The notebook downloads the dataset using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "sachinkumar413/covid-pneumonia-normal-chest-xray-images"
)
```

The dataset identifier is:

```text
sachinkumar413/covid-pneumonia-normal-chest-xray-images
```

The dataset contains chest X-ray images associated with COVID, pneumonia, and normal cases. The dataset itself is not redistributed with this repository. Users should obtain it from the original source and comply with its terms of use.

**Original dataset source:**  
https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images

The dataset source should also be cited in the manuscript's Materials and Methods section as required by the journal.

## Code Information

The main implementation is provided as a Jupyter Notebook.

The code performs the following operations:

1. Installs/imports the required Python packages.
2. Downloads and prepares the chest X-ray dataset.
3. Loads and preprocesses an X-ray image.
4. Loads a pretrained ResNet-50 model.
5. Generates adversarial examples using FGSM, PGD, and DeepFool.
6. Generates Grad-CAM activation maps from multiple ResNet-50 layers.
7. Combines layer-wise activation maps into an LPAM representation.
8. Calculates activation-based metrics.
9. Applies an attack-detection threshold.
10. Visualizes the original image, adversarial image, layer-wise CAMs, and LPAM output.

## Methodology

### Image Preprocessing

Input chest X-ray images are:
- Read using OpenCV.
- Converted from BGR to RGB.
- Resized to 224 × 224 pixels.
- Converted to floating-point values in the range [0, 1].
- Normalized using the ImageNet mean and standard deviation:

```text
Mean = [0.485, 0.456, 0.406]
Standard deviation = [0.229, 0.224, 0.225]
```

### ResNet-50

The implementation uses the pretrained ResNet-50 model provided by Torchvision:

```python
model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)
```

The model is used in evaluation mode. CUDA is used when available.

### FGSM Attack

The Fast Gradient Sign Method is implemented using the gradient of the classification loss with respect to the input image.

The implementation uses:

```text
epsilon = 8 / 255
```

### PGD Attack

Projected Gradient Descent is implemented as an iterative attack using:

```text
epsilon = 32 / 255
alpha = 4 / 255
iterations = 20
```

After each update, the perturbation is projected into the specified epsilon neighborhood around the original input.

### DeepFool Attack

A custom DeepFool implementation is included.

The implementation uses:

```text
num_classes = 10
overshoot = 0.02
maximum iterations = 50
```

The attack iteratively estimates a perturbation that can move the input across a model decision boundary.

## Layer-wise Progressive Activation Mapping (LPAM)

Grad-CAM activation maps are generated from the following ResNet-50 layers:

```python
model.layer1[-1]
model.layer2[-1]
model.layer3[-1]
model.layer4[-1]
```

For each layer:
1. Grad-CAM is calculated for the predicted class.
2. The activation map is obtained.
3. The mean activation is calculated.
4. The layer maps are combined using normalized mean-activation weights.

The resulting weighted activation maps form the LPAM representation, which is normalized between 0 and 1.

## Activation-Based Metrics

The implementation calculates:

### Entropy

The normalized LPAM activation is treated as a distribution and entropy is calculated as:

```text
Entropy = -Σ p log(p)
```

### Activation Area

The number of pixels with LPAM activation greater than 0.6 is calculated.

### Area Ratio

```text
Area Ratio = Activation Area / Total LPAM Pixels
```

### Maximum Activation

The maximum value of the normalized LPAM is recorded.

### Concentration Score

```text
Concentration Score =
Maximum Activation / Mean Activation
```

### Attack Score

The implementation calculates:

```text
Attack Score =
    0.4 × (1 / Entropy)
  + 0.3 × Concentration Score
  + 0.3 × (1 - Area Ratio)
```

Small numerical constants used in the notebook prevent division by zero.

## Attack Detection

The notebook uses:

```text
THRESHOLD = 1.35
```

The implemented classification rule is:

```python
if metrics["Attack Score"] > THRESHOLD:
    print("Prediction : Adversarial Attack Detected")
else:
    print("Prediction : Clean Image")
```

The threshold is part of the current experimental implementation and can be modified for further experiments.

## Visualization

The notebook generates visualizations containing:
1. Original image
2. Processed/adversarial image
3. Grad-CAM from ResNet-50 Layer 1
4. Grad-CAM from ResNet-50 Layer 2
5. Grad-CAM from ResNet-50 Layer 3
6. Grad-CAM from ResNet-50 Layer 4
7. LPAM visualization

## Requirements

The implementation requires Python and the following packages:

```text
torch
torchvision
grad-cam
opencv-python
matplotlib
scikit-image
numpy
kagglehub
```

Install them with:

```bash
pip install torch torchvision grad-cam opencv-python matplotlib scikit-image numpy kagglehub
```

A CUDA-capable GPU is recommended for faster execution, although the implementation can run on CPU.

## Usage Instructions

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Install dependencies

```bash
pip install torch torchvision grad-cam opencv-python matplotlib scikit-image numpy kagglehub
```

### 3. Obtain the dataset

Download the dataset from the original Kaggle source or use the KaggleHub code provided in the notebook.

### 4. Set the dataset path

The notebook currently uses paths such as:

```text
/content/datasets
```

and an example image path such as:

```text
/content/datasets/COVID/COVID_1002.png
```

Modify these paths when running the notebook outside Google Colab.

### 5. Run the notebook

Open the Jupyter Notebook and execute the cells in order.

The notebook will:
- Load the dataset
- Preprocess an X-ray image
- Load ResNet-50
- Generate adversarial examples
- Generate Grad-CAM maps
- Construct LPAM
- Calculate attack metrics
- Classify the image using the configured threshold
- Display the resulting visualizations

## Reproducibility

To reproduce the implementation:

1. Use a compatible Python environment.
2. Install all dependencies listed above.
3. Obtain the original chest X-ray dataset.
4. Use the same ResNet-50 pretrained weights.
5. Use the same preprocessing parameters.
6. Use the same FGSM, PGD, and DeepFool parameters.
7. Use the same ResNet-50 layers for Grad-CAM.
8. Use the same LPAM calculation.
9. Use the same metric definitions.
10. Use the configured threshold of 1.35.

Exact numerical results can depend on software versions, model weights, hardware, and numerical precision.

## Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── notebooks/
│   └── adversarial_attack_detection.ipynb
├── data/
│   └── README.md
├── results/
│   └── README.md
└── LICENSE
```

The chest X-ray dataset should not be committed to the repository unless its license explicitly permits redistribution.

## Citations

If this repository or its implementation is used in academic work, please cite the associated research article.

### Dataset

COVID-Pneumonia-Normal Chest X-Ray Images.

Kaggle dataset identifier:

```text
sachinkumar413/covid-pneumonia-normal-chest-xray-images
```

Dataset URL:

https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images

### Software

This project uses:
- PyTorch
- Torchvision
- PyTorch Grad-CAM
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- KaggleHub

Please consult the respective projects for their licensing and citation requirements.

## Data and Code Availability

The source notebook and documentation are provided to support reproducibility of the associated research.

The chest X-ray dataset is not included in this repository. Users should download it from the original source.

## License

Add the project's selected license as a `LICENSE` file before public release.

Third-party datasets and software remain subject to their respective licenses and terms of use.

## Contribution Guidelines

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Document any changes to the implementation or methodology.
4. Test the modified code.
5. Submit a pull request describing the changes.

Changes affecting experimental results should include enough information to support reproducibility.

## Disclaimer

This repository contains research code for studying adversarial perturbations and activation patterns in chest X-ray images.

It is intended for research and experimental purposes and is **not a clinical diagnostic system**. It should not be used to make medical decisions.

## Contact

For questions regarding the implementation or associated research, please contact the corresponding author(s) of the associated manuscript.
