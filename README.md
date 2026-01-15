# Eye Disease Classification from Retinal Images

This repository presents a complete deep learning pipeline for **classifying eye diseases** from retinal images.  
The project explores and compares a **custom CNN model** with **pretrained EfficientNet architectures (B0 and B3)**, evaluating accuracy, efficiency, and suitability for deployment.


---

##  The workflow includes:

- Dataset validation and cleaning
- Train / validation / test splitting
- Data loading using **TensorFlow**
- Data augmentation using **Keras preprocessing layers**
- Training from scratch (CNN) and transfer learning (EfficientNet)
- Fine-tuning pretrained models
- Model evaluation with **accuracy, classification report, and confusion matrix**
- Saving trained models and class labels
- Deployment-ready inference using **Gradio**

---

## Dataset

This project uses the **eye_diseases_classification** on Kaggle:  
[eye_diseases_classification]([https://www.kaggle.com/datasets/gunavenkatdoddi/eye-disease-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

**Important:** The dataset is **not included** in this repository due to its size.  
1. Download the dataset from Kaggle.  
2. Extract and place it in your working environment, e.g., `/kaggle/working/dataset/`.  
3. The training notebook expects a folder structure:
---
   ```bash
dataset/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/

```



## 📁Project Structure

```bash
Eye-Disease-Classification/
│
├── README.md
├── requirements.txt      # Dependencies
├── notebooks/
│ ├── eye-diseases-cnn-vs-efficientnet-vs-fine-tuning.ipynb # Data preprocessing & training
├── models/               # Saved trained models
│ ├── eye_cnn_model.h5
│ ├── eye_efficient_b0_modell.h5
│ ├── eye_efficient_b3_model.h5
│ └── finetuned_model.h5
│ └── CLASS_NAMES.json 
├── src/
│ ├── deployment.py
└── assets/
│ ├──examples.png  



```

### Setup Instructions

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```


### Training Models
```bash
Open notebooks/eye-diseases-cnn-vs-efficientnet-vs-fine-tuning.ipynb.

Ensure the dataset path is correctly set.

Run cells sequentially to:

Load and verify images

Split dataset

Apply augmentation

Train CNN and EfficientNet (B0 & B3)

Evaluate models and save .h5 files and class_names.json

```

### Deployment (Gradio)
```bash

Open srcs/deployment.py

Load a trained model and class_names.json.

Run Gradio interface to upload retinal images and predict eye disease.

```

### Data Preprocessing & Augmentation

Images resized to match the model input (224x224 for CNN/EfficientNetB0, 300x300 for EfficientNetB3)

Augmentation applied: random horizontal flips, rotation, zoom, and contrast adjustment

Scaling / normalization applied using TensorFlow or model-specific preprocessing (preprocess_input for EfficientNet)

### Notes 

EfficientNetB0 is faster and lighter for deployment, especially with Gradio.

EfficientNetB3 may provide higher accuracy but is heavier and slower.

Models and class labels are saved in models/ for inference

Trained Models

The trained models are stored in compressed format (.rar) due to their large size.
After downloading and extracting the archive, place the models in the models/ folder:

 ```bash

models/
├── cnn_model.h5
├── efficientb0_model.h5
├── efficientb3_model.h5
├── finetuned_model.h5
├── class_names.json

Ensure the filenames match exactly for the deployment notebook or Gradio app to work.
You can download the .rar file from the repository releases section.
```

---



