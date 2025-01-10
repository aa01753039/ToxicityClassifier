# ToxicityClassifier

This repository contains code and resources for building a robust toxicity classification system. The project explores the use of generative and discriminative models and combines them to enhance classification accuracy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Generative Model](#generative-model)
  - [Discriminative Model](#discriminative-model)
  - [Model Combination](#model-combination)
- [Performance Evaluation](#performance-evaluation)
- [Explainable AI](#explainable-ai)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project aims to classify toxic comments using state-of-the-art machine learning models. It leverages:
- **Generative models** (GPT-2) for understanding text context.
- **Discriminative models** (BERT) for precise classification.
- A **combined model** approach to maximize classification performance.

---

## Installation

Ensure you have the required libraries installed. Use the following commands:

```bash
pip install transformers
pip install numpy pandas scikit-learn matplotlib shap
pip install gdown pydrive
```

---

## Usage

### Running the Project

The project operates in multiple stages:

1. **Train Models**:
   - Train the **Generative Model**:
     ```bash
     Run main() > Option 1
     ```
   - Train the **Discriminative Model**:
     ```bash
     Run main() > Option 3
     ```
   - Train the **Combined Model**:
     ```bash
     Run main() > Option 5
     ```

2. **Test Models**:
   - Test the **Generative Model**:
     ```bash
     Run main() > Option 2
     ```
   - Test the **Discriminative Model**:
     ```bash
     Run main() > Option 4
     ```
   - Test the **Combined Model**:
     ```bash
     Run main() > Option 6
     ```

3. **Classify Text**:
   Use the combined model to classify an input text:
   ```bash
   Run main() > Option 7
   ```

---

## Code Explanation

### Generative Model

- **Model Used**: GPT-2
- **Pipeline**:
  - Data preprocessing
  - Training using the training dataset
  - Validation and saving the best model
- **Key Outputs**:
  - Trained GPT-2 model saved in `Model_Gen/`
  - Predictions on the test dataset

### Discriminative Model

- **Model Used**: BERT
- **Pipeline**:
  - Data tokenization
  - Model fine-tuning on labeled data
  - Validation and saving the best model
- **Key Outputs**:
  - Trained BERT model saved in `Model_Dis/`
  - Predictions on the test dataset

### Model Combination

- **Technique**: Stacking
  - Combines the predictions from GPT-2 and BERT using a Logistic Regression model.
  - Predictions are weighted based on the strengths of each model.
- **Key Outputs**:
  - Combined model saved in `Model_Combine/`
  - Enhanced predictions leveraging both generative and discriminative models

---

## Performance Evaluation

- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Performance is evaluated and visualized for:
  - Generative Model
  - Discriminative Model
  - Combined Model

---

## Explainable AI

- **SHAP** (SHapley Additive exPlanations) is used to interpret model predictions.
- Visualizes feature importance for both the generative (GPT-2) and discriminative (BERT) models.
- To generate explanations:
  ```bash
  explainable_ai(text, model_gen_gdrive_link, token_gen_gdrive_link, model_dis_gdrive_link, token_dis_gdrive_link)
  ```

---

## Acknowledgements

This project leverages various open-source tools and libraries, including:
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SHAP](https://shap.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Google Colab](https://colab.research.google.com/) for training and experimentation.
