🚀 Overview

This project builds, evaluates, and exports a robust fake news detection model using both classic machine learning (Logistic Regression, SVM, Naive Bayes) and advanced deep learning (DistilBERT via Hugging Face Transformers).

Key Steps:

Data exploration & EDA (pandas, scikit-learn, matplotlib, seaborn)

Text preprocessing (NLTK, regex, emoji)

Classic ML (Logistic Regression, SVM, Naive Bayes)

LLM Fine-tuning (DistilBERT with Hugging Face)

Model export & deployment (PyTorch, ONNX, ONNX Runtime)

📂 Dataset

Source:
Fake news datasets from Kaggle or custom sources.

Fake.csv: Labeled fake news articles

True.csv: Labeled true news articles

Both files must include title and text columns.

📊 Project Steps

Data Loading & EDA:

Load, shuffle, and explore data.

Visualize class balance and document statistics.

Text Preprocessing:

Clean text (lowercase, remove punctuation, digits, HTML, emoji)

Tokenization, stopword removal, stemming

Feature Engineering:

TF-IDF, n-grams, word embeddings (Word2Vec)

Feature importance exploration

Classic ML:

Train/test split (stratified)

Train and evaluate Logistic Regression, Naive Bayes, SVM

Cross-validation

Transformers Fine-tuning:

Prepare Hugging Face datasets.Dataset

Tokenize and fine-tune DistilBERT for fake news detection

Custom metrics, confusion matrix, error analysis

Model Export & Serving:

Save model & tokenizer (Hugging Face)

Export to ONNX

Predict using ONNX Runtime

🛠️ Requirements

Python 3.8+

pandas, scikit-learn, matplotlib, seaborn

nltk, gensim, emoji

torch, transformers, datasets, evaluate

onnx, onnxruntime

Install requirements with:

pip install -r requirements.txt

🏃‍♂️ How to Run
1. Train & Fine-tune

Run the notebook fake_news_detection.ipynb (Colab, Kaggle, or Jupyter)

Follow the stages step by step, from EDA to deployment.

2. Serve or Export the Model

The fine-tuned model is saved to distilbert_fake_news_final/

ONNX model is exported as distilbert_fake_news.onnx

Inference Example:

PyTorch (Hugging Face)

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
model = DistilBertForSequenceClassification.from_pretrained("distilbert_fake_news_final")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_fake_news_final")
texts = ["Sample headline...", "Another news text..."]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
with torch.no_grad():
    preds = model(**inputs).logits.argmax(dim=1)
print(preds)  # 0 = Fake, 1 = True


ONNX Runtime

import onnxruntime as ort
sess = ort.InferenceSession("distilbert_fake_news.onnx")
# Use the same tokenized inputs as above
onnx_inputs = {
    "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int64),
    "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int64)
}
onnx_preds = sess.run(None, onnx_inputs)[0].argmax(axis=1)
print(onnx_preds)

🙋‍♂️ Author

Ghassan Alkahlout
