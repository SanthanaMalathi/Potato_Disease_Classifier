# Potato Leaf Disease Classification Using Deep Learning

An end-to-end deep learning project that classifies potato leaf images into **Early Blight, Late Blight, and Healthy** categories using a fine-tuned **EfficientNetB0** model and provides an interactive **Streamlit dashboard** for image-based prediction.

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and quality. Identifying diseases manually from leaf images can be time-consuming and may require domain expertise.

This project develops an image classification system that can automatically classify potato leaf images into three categories:

* Potato Early Blight
* Potato Late Blight
* Potato Healthy

The project follows an end-to-end machine learning workflow, from dataset exploration and preprocessing to model development, transfer learning, fine-tuning, evaluation, model saving, and deployment through Streamlit.

---

## 🎯 Objectives

* Build a deep learning model for potato leaf disease classification.
* Analyze the image dataset and class distribution.
* Create stratified training, validation, and test datasets.
* Develop a baseline CNN model.
* Apply transfer learning using EfficientNetB0 pretrained on ImageNet.
* Fine-tune the pretrained model for potato leaf classification.
* Evaluate model performance using accuracy, classification report, and confusion matrix.
* Save the trained model for inference.
* Build an interactive Streamlit dashboard for image-based predictions.

---

## 🗂️ Dataset

The dataset contains images belonging to three potato leaf categories.

| Class        |    Images |
| ------------ | --------: |
| Early Blight |     1,000 |
| Healthy      |       152 |
| Late Blight  |     1,000 |
| **Total**    | **2,152** |

The dataset is divided using stratified sampling into:

* **80% Training**
* **10% Validation**
* **10% Testing**

Stratification is used to preserve the class distribution across the training, validation, and test sets.

---

## 🔍 Exploratory Data Analysis

The project begins by examining the dataset structure and class distribution.

The class distribution shows that the Healthy class contains substantially fewer images than the Early Blight and Late Blight classes.

To account for the class imbalance during model training, class weights are incorporated into the training process.

---

## 🖼️ Image Preprocessing

Images are processed using a TensorFlow `tf.data` pipeline.

### Preprocessing steps

1. Load image files from their respective class directories.
2. Decode images using TensorFlow.
3. Convert images to 3-channel RGB.
4. Resize images to **224 × 224 pixels**.
5. Convert image values to floating-point representation.
6. Batch images using a batch size of **32**.
7. Prefetch batches using TensorFlow AUTOTUNE.

The same input size is used for both the baseline CNN and EfficientNetB0.

---

## 🔄 Data Augmentation

Data augmentation is applied during training to improve model generalization and reduce overfitting.

The augmentation pipeline is used before feeding images into the models.

This helps the model learn more robust visual patterns rather than relying heavily on the exact orientation or appearance of individual training images.

---

# 🧠 Model Development

Two major approaches were evaluated.

## 1. Baseline CNN

A custom Convolutional Neural Network was developed as a baseline.

The architecture includes:

* Image rescaling
* Data augmentation
* Convolutional layers
* Max pooling
* Dense layers
* Dropout regularization
* Softmax output layer

The model was trained using:

* Optimizer: Adam
* Loss: Sparse Categorical Cross-Entropy
* Metric: Accuracy
* Epochs: 30

The baseline CNN achieved:

**Test Accuracy: 95.83%**

---

# ⚡ 2. Transfer Learning with EfficientNetB0

To improve the baseline performance, transfer learning was applied using **EfficientNetB0 pretrained on ImageNet**.

The pretrained convolutional backbone was initially frozen and used as a feature extractor.

A custom classification head was added consisting of:

* EfficientNetB0 backbone
* Global Average Pooling
* Dropout
* Dense Softmax classification layer

The model was initially trained with a learning rate of `0.001`.

The EfficientNetB0 transfer learning model achieved:

**Test Accuracy: 97.69%**

---

# 🔧 Fine-Tuning

After training the classification head, the final layers of EfficientNetB0 were selectively unfrozen.

The earlier layers remained frozen to preserve generic visual features, while later layers were allowed to adapt to potato leaf-specific visual patterns.

A smaller learning rate of:

`1e-5`

was used during fine-tuning.

This helped improve the model's ability to learn disease-specific image features.

---

# 📊 Model Performance

The three approaches were compared using the test dataset.

| Model                            | Test Accuracy |
| -------------------------------- | ------------: |
| Baseline CNN                     |        95.83% |
| EfficientNetB0 Transfer Learning |        97.69% |
| Fine-Tuned EfficientNetB0        |    **99.07%** |

The fine-tuned EfficientNetB0 was selected as the final model.

### Final Test Accuracy

**99.07%**

The final model is evaluated further using:

* Precision
* Recall
* F1-score
* Confusion Matrix
* Class-wise performance

---

# 📈 Evaluation

The final model is evaluated on an unseen test dataset.

Evaluation includes:

### Classification Report

The classification report provides class-wise:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

The confusion matrix is used to analyze:

* Correct predictions
* Misclassified images
* Class-specific weaknesses
* Confusion between disease categories

---

# 🔎 Prediction Pipeline

The trained model can accept an input leaf image and generate a prediction.

The inference pipeline:

```text
Input Image
     ↓
Resize to 224 × 224
     ↓
Convert to Tensor
     ↓
Add Batch Dimension
     ↓
EfficientNetB0 Model
     ↓
Softmax Probabilities
     ↓
Highest Probability Class
     ↓
Prediction + Confidence
```

A confidence threshold is also applied during inference.

If the highest prediction probability is below **60%**, the application returns:

```text
Uncertain
```

This prevents the application from presenting a low-confidence prediction as a definitive classification.

---

# 🖥️ Streamlit Dashboard

An interactive Streamlit dashboard was developed to make the trained model accessible through a simple web interface.

### Dashboard workflow

```text
Upload Potato Leaf Image
          ↓
Image Preview
          ↓
Model Prediction
          ↓
Predicted Class
          ↓
Confidence Score
```

The dashboard allows users to upload an image and receive the predicted potato leaf condition along with the model confidence.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* EfficientNetB0
* Transfer Learning
* Fine-Tuning

### Machine Learning / Evaluation

* Scikit-learn
* Classification Report
* Confusion Matrix

### Data & Image Processing

* NumPy
* TensorFlow Image Processing

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit
* Streamlit Community Cloud

### Development Environment

* Jupyter Notebook

---

# 📁 Project Structure

```text
potato-leaf-disease-classifier/
│
├── app.py
├── potato_disease_model.keras
├── class_names.json
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebook/
│   └── potato_plant_Disease_classifier.ipynb
│
└── screenshots/
    ├── home.png
    ├── prediction.png
    └── confusion_matrix.png
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/potato-leaf-disease-classifier.git
```

Move into the project directory:

```bash
cd potato-leaf-disease-classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
streamlit
tensorflow
numpy
pillow
scikit-learn
matplotlib
seaborn
```

Use the Python/TensorFlow versions that you have actually tested locally when deploying.

---

# ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

Upload a potato leaf image to generate a prediction.

---

# 💾 Model

The final fine-tuned EfficientNetB0 model is saved in Keras format:

```text
potato_disease_model.keras
```

The model is loaded by the Streamlit application and used directly for inference without retraining.

---

# 🧪 Model Development Workflow

```text
Dataset
   ↓
EDA & Class Distribution
   ↓
Stratified Train / Validation / Test Split
   ↓
TensorFlow Data Pipeline
   ↓
Image Resizing & Augmentation
   ↓
Baseline CNN
   ↓
Baseline Evaluation
   ↓
EfficientNetB0 Transfer Learning
   ↓
Transfer Learning Evaluation
   ↓
Fine-Tuning
   ↓
Final Evaluation
   ↓
Save Model
   ↓
Streamlit Dashboard
   ↓
Deployment
```

---

# 📌 Key Results

* Built an end-to-end potato leaf disease classification pipeline.
* Classified images into three categories: Early Blight, Late Blight, and Healthy.
* Developed a custom CNN baseline achieving **95.83% test accuracy**.
* Improved performance to **97.69%** using EfficientNetB0 transfer learning.
* Achieved **99.07% test accuracy** after fine-tuning EfficientNetB0.
* Implemented class-weighted training to address class imbalance.
* Added confidence-based inference with an uncertainty threshold.
* Saved the trained model for deployment.
* Developed an interactive Streamlit dashboard for real-time image prediction.

---

# ⚠️ Limitations

The model is trained on a specific dataset of potato leaf images. Real-world images may differ in:

* Lighting conditions
* Backgrounds
* Camera quality
* Leaf orientation
* Disease severity
* Image resolution
* Environmental conditions

Therefore, the model should be treated as a classification assistance tool rather than a replacement for professional agricultural diagnosis.

---

# 🔮 Future Improvements

Possible improvements include:

* Expand the dataset with real-world field images.
* Add more potato disease categories.
* Improve robustness to different lighting and backgrounds.
* Add Grad-CAM visualizations to explain model predictions.
* Experiment with additional pretrained architectures.
* Add prediction history to the Streamlit dashboard.
* Add batch image prediction.
* Add model monitoring after deployment.

---

## ⭐ Project Highlights

**Deep Learning | Computer Vision | Image Classification | Transfer Learning | EfficientNetB0 | Fine-Tuning | TensorFlow | Keras | Streamlit | Model Deployment**
