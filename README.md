# Disease Detection from Chest X-Ray 🩻🧠

## Project Goal
Build a deep learning model to classify chest X-ray images as **Normal** or **Pneumonia**, demonstrating how AI can support medical diagnosis.

---

## Dataset
**Chest X-Ray Images (Pneumonia)** – [Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

- Total images: 5,863 across training, test, and validation sets  
- Classes: `NORMAL`, `PNEUMONIA`  
- Class distribution in training set:
  - Normal: 1,342 images  
  - Pneumonia: 3,876 images  

> Note: The dataset is imbalanced, with ~3x more pneumonia cases than normal.

---

## Installation & Setup

1. Clone the repository:
```bash
git clone <https://github.com/riadkassem/Pneumonia-Disease-Detection-From-Chest-X-ray.git>
cd <DiseaseDetection>
```
2. Install required libraries:
```bash
pip install tensorflow matplotlib numpy seaborn sklearn kagglehub
```
3. Set up Kaggle dataset (replace with your Kaggle credentials if needed):
```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```
## Data Preparation
- Images are resized to (150, 150)
 
- Batch size: 32

- Normalization applied to scale pixel values to [0, 1]

- Data augmentation applied: random horizontal flip, rotation, and zoom

## Model Architecture
- Input layer: (150, 150, 3)

- Data augmentation layer

- Conv2D + MaxPooling blocks:

    - 32 filters → 64 filters → 128 filters

- Flatten → Dense(64) → Dense(1, activation='sigmoid')

- Binary classification: Normal (0) / Pneumonia (1)

## Training
- Optimizer: Adam

- Loss: Binary Crossentropy

- Metrics: Accuracy

- Epochs: 10 (adjustable)

## Evaluation
- Classification report and confusion matrix generated using test set

- Current performance:

    - Recall (Pneumonia): 81%

    - Precision (Pneumonia): 62%

    - Recall (Normal): 18%

    - Overall Accuracy: 57%

- The model is not yet suitable for clinical deployment. Improvements can include:

    - Addressing class imbalance

    - Refining architecture

    - Adjusting decision thresholds

## Visualizations
- Sample images from training dataset

- Training/validation accuracy and loss curves

- Confusion matrix heatmap

## Usage
```python
# Example: Predict on a new chest X-ray image
from tensorflow.keras.preprocessing import image
img = image.load_img('path_to_image.jpg', target_size=(150,150))
img_array = tf.expand_dims(tf.keras.utils.img_to_array(img)/255.0, 0)
prediction = model.predict(img_array)
if prediction > 0.5:
    print("Pneumonia")
else:
    print("Normal")
```