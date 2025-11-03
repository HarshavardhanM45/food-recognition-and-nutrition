# 🍛 Indian Food Classification & Nutrition Analysis

## 📘 Abstract
The **Indian Food Classification and Nutrition Analysis** project is an AI-based system that classifies Indian food items from images using **Deep Learning (CNN)** and provides **nutritional information** like calories, protein, fat, and carbohydrates.  
It combines **image processing**, **machine learning**, and **data analytics** to help users monitor their diet and make healthier choices.

---

## 🎯 Project Aim
To design and develop a deep learning-based model capable of identifying Indian food from an image and estimating its nutritional composition using a predefined dataset.

---

## 🧠 Problem Statement
Manual food identification and calorie calculation are time-consuming, inaccurate, and impractical in real-world applications like diet tracking and restaurant meal logging.  
This project provides an **automated and intelligent solution** to predict food items and estimate nutritional values directly from an image.

---

## 🧩 Objectives
- Build a Convolutional Neural Network (CNN) to classify Indian food images.  
- Train the model using a labeled dataset of Indian dishes.  
- Retrieve nutritional information from a CSV dataset based on the predicted food name.  
- Summarize total nutrition for a meal consisting of multiple food items.  
- Provide an extendable structure for integrating into web or mobile apps.

---

## ⚙️ Technologies Used

### 🧮 Machine Learning & Deep Learning
- **TensorFlow / Keras:** Building and training the CNN model  
- **NumPy:** Numerical computations and array manipulation  
- **Pandas:** Data handling for nutrition dataset  
- **Matplotlib:** Visualization of training and accuracy metrics  
- **Pickle:** Saving model class mappings and nutrition data  

### 💻 Tools & Environment
- **Python 3.8+**
- **Jupyter Notebook / VS Code**
- **Git & GitHub**
- **OS Compatibility:** Windows / Linux / macOS  

---

## 🧱 Project Structure
Indian-Food-Classification/
│
├── main.py # Main Python file (model + nutrition logic)
├── nutrition.csv # Nutrition dataset
├── requirements.txt # Required dependencies
├── README.md # Full project documentation
├── class_indices.pkl # Saved class labels
├── nutrition.pkl # Saved nutrition dataframe
│
└── Indian Food Images/ # Dataset folder (not uploaded)
├── aloo_gobi/
├── biryani/
├── dosa/
└── samosa/

> ⚠️ Note: The trained model file (`food_cnn_model.h5`) is not uploaded due to file size restrictions on GitHub.  
> You can train your own model using `main.py`.

---

## 📊 Dataset Description

### 📁 Image Dataset
The image dataset contains categorized folders of Indian food items.  
Each folder represents a unique class.  

**Example Folder Structure:**
Indian Food Images/
├── aloo_gobi/
│ ├── 1.jpg
│ ├── 2.jpg
├── biryani/
├── dosa/
└── samosa/
Each folder includes 50–100 images of the respective dish collected from online sources or open datasets.  
Images are resized to **128×128 pixels** for uniform input size to the CNN model.

---

### 📄 Nutrition Dataset
A simple **CSV file** named `nutrition.csv` is used to store nutrition data for each food item.  
This dataset maps every food label to its nutritional content per serving.

**Example:**
| Food Name | Calories | Protein | Fat | Carbs |
|------------|-----------|----------|------|--------|
| aloo_gobi  | 150 | 3 | 5 | 20 |
| biryani    | 300 | 10 | 8 | 40 |
| dosa       | 180 | 4 | 3 | 25 |
| samosa     | 250 | 5 | 10 | 30 |

These values are estimated averages per 100g serving, sourced from public nutrition databases.

---

## 🧠 Model Description — CNN (Convolutional Neural Network)

The **Convolutional Neural Network (CNN)** is a type of deep learning model designed for image recognition.  
It automatically extracts important features like shapes, textures, and colors from food images to classify them accurately.

### 🧩 Architecture
| Layer Type | Configuration | Function |
|-------------|---------------|-----------|
| Input | 128×128×3 | Input image (RGB) |
| Conv2D + ReLU | 32 filters, 3×3 kernel | Detects low-level features |
| MaxPooling2D | 2×2 | Reduces image size |
| Conv2D + ReLU | 64 filters, 3×3 kernel | Extracts complex features |
| MaxPooling2D | 2×2 | Controls overfitting |
| Flatten | — | Converts 2D matrix to 1D |
| Dense (128 units) | ReLU | Learns non-linear patterns |
| Dense (n classes) | Softmax | Outputs prediction probabilities |

**Optimizer:** `adam`  
**Loss Function:** `categorical_crossentropy`  
**Metric:** `accuracy`  

---

## ⚙️ Model Training Workflow

1. **Data Loading:** Load images and labels using `ImageDataGenerator`.  
2. **Preprocessing:** Resize images, normalize pixel values (0–1).  
3. **Augmentation:** Random flips, rotations, and zooms for generalization.  
4. **Model Building:** Define CNN using Keras Sequential API.  
5. **Compilation:** Use Adam optimizer and categorical loss.  
6. **Training:** Train on 80% training data, validate on 20%.  
7. **Evaluation:** Plot accuracy and loss graphs.  
8. **Saving:** Export model (`.h5`) and mappings (`.pkl`) for future use.

---

## 🧾 Output Explanation

### ✅ Single Image Prediction
```bash
Predicted Food: biryani

Nutritional Values:
Calories: 300
Protein: 10g
Fat: 8g
Carbs: 40g
| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 94%   |
| Validation Accuracy | 89%   |
| Loss                | 0.21  |
| Epochs              | 25    |
