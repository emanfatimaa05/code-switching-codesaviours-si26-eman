Code Switching NLP | Code Saviours SI-26 | Eman Fatima

# Roman Urdu–English Code-Switching NLP

A Natural Language Processing project that identifies whether individual words in Roman Urdu–English mixed text are **Urdu (URD)** or **English (ENG)**.

## 📌 Why This Matters

Roman Urdu is widely used in everyday communication in Pakistan, and it is often mixed with English in the same sentence. This code-switching makes language processing more challenging for NLP systems.

This project focuses on identifying Urdu and English words within mixed-language Roman Urdu text. This can support language analysis and other NLP applications involving Roman Urdu and English.

## 🎯 Project Objective

The goal of this project was to:

- Create a Roman Urdu–English code-switching dataset
- Label individual words as Urdu or English
- Train a language identification model
- Evaluate the model on unseen test sentences
- Save and publish the trained model on Hugging Face

## 📊 Dataset

The final dataset contains:

- **157 unique sentences**
- **1,593 total word entries**
- **1,035 URD-labelled words**
- **558 ENG-labelled words**
- **125 training sentences**
- **32 testing sentences**

Each word in the dataset is labelled as either:

- `URD` — Roman Urdu
- `ENG` — English

The dataset is available in the repository as:

`dataset.csv`

## ⚙️ How It Works

First, Roman Urdu–English sentences were collected and converted into a labelled dataset. Each word was assigned either an `URD` or `ENG` label.

The sentences were then divided into training and testing sets. A pretrained **XLM-RoBERTa** model was fine-tuned for token classification so that it could identify the language of individual words.

After training, the model was evaluated on the testing sentences and the final model was uploaded to Hugging Face.

## 🧠 Model

**Model:** XLM-RoBERTa (`xlm-roberta-base`)

**Task:** Roman Urdu–English Token Classification

**Labels:**

- URD
- ENG

The model was trained for **5 epochs** using the Hugging Face Transformers Trainer.

## 📈 Results

The final model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | **90.91%** |
| URD F1 Score | **92.91%** |
| ENG F1 Score | **87.34%** |
| Overall F1 Score | **91.13%** |

The model performed particularly well at identifying Urdu words, while also achieving strong performance on English words.

**MIX was not applicable** because the final dataset did not contain any MIX-labelled examples.

## 🤗 Hugging Face Model

The trained model is published on Hugging Face:

**Model:** https://huggingface.co/emanfatimaa05/code-switching-codesaviours-si26-eman

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Hugging Face Datasets
- scikit-learn
- pandas
- seqeval
- Google Colab
- Google Drive
- Hugging Face Hub

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/emanfatimaa05/code-switching-codesaviours-si26-eman.git
cd code-switching-codesaviours-si26-eman
````

### 2. Install the required libraries

```bash
pip install torch transformers datasets seqeval accelerate scikit-learn pandas
```

### 3. Load the trained model

The trained model can be loaded directly from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "emanfatimaa05/code-switching-codesaviours-si26-eman"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
```

### 4. Classify Roman Urdu–English text

```python
import torch

text = "Aaj mera mood nahi hai for anything"
words = text.split()

inputs = tokenizer(
    words,
    is_split_into_words=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

predictions = outputs.logits.argmax(dim=-1)

for word, prediction in zip(words, predictions[0][1:-1]):
    label = model.config.id2label[prediction.item()]
    print(f"{word}: {label}")
```

The model predicts whether each word belongs to the `URD` or `ENG` category.

## 📁 Repository Structure

```text
code-switching-codesaviours-si26-eman/
│
├── dataset.csv
├── SI26_Week6_eman.ipynb
├── SI26_Week7_Eman.ipynb
└── README.md
```

### 📓 Week 6 — Dataset Collection

The Week 6 notebook contains the dataset collection, preparation, word-level labelling, validation, and creation of the final `dataset.csv`.

### 📓 Week 7 — Model Training & Evaluation

The Week 7 notebook contains dataset loading, train/test splitting, XLM-RoBERTa setup, tokenization, model training, evaluation, model saving, and Hugging Face upload.

## 🎥 Demo

A demo video will be added here:

**Loom Demo:** <https://www.loom.com/share/5476430174334c4e9a53f971b394e9e3>

## 🚀 Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset
* Adding more naturally occurring code-switched examples
* Supporting additional language categories
* Improving performance on ambiguous words
* Building an interactive interface for real-time language identification

## 🎓 Internship Project

This project was developed as part of the **Code Saviours SI-26 Machine Learning / AI Internship Programme — 2026**.

The project provided practical experience in dataset creation, NLP preprocessing, transformer-based model training, evaluation, model deployment, and version control.

## 👩‍💻 Built By

**Eman Fatima**
**Code Saviours SI-26 | 2026**
```
