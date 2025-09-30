Fake News Detection: ML & Transformers

🚀 Overview

This project builds, evaluates, and exports a robust fake news detection model using both classic machine learning (Logistic Regression, SVM, Naive Bayes) and advanced deep learning (DistilBERT via Hugging Face Transformers).




Key Steps:

1-Data exploration & EDA (pandas, scikit-learn, matplotlib, seaborn)

2-Text preprocessing (NLTK, regex, emoji)

3-Classic ML (Logistic Regression, SVM, Naive Bayes)

4-LLM Fine-tuning (DistilBERT with Hugging Face)

5-Model export & deployment (PyTorch, ONNX, ONNX Runtime)





📂 Dataset

Source:

Fake news datasets from Kaggle or custom sources.

Fake.csv: Labeled fake news articles

True.csv: Labeled true news articles

Both files must include title and text columns.


📊 Project Steps

1-Data Loading & EDA:

Load, shuffle, and explore data.

Visualize class balance and document statistics.

2-Text Preprocessing:

Clean text (lowercase, remove punctuation, digits, HTML, emoji)

Tokenization, stopword removal, stemming

Feature Engineering:

TF-IDF, n-grams, word embeddings (Word2Vec)

Feature importance exploration

3-Classic ML:

Train/test split (stratified)

Train and evaluate Logistic Regression, Naive Bayes, SVM

Cross-validation

Transformers Fine-tuning:

Prepare Hugging Face datasets.Dataset

Tokenize and fine-tune DistilBERT for fake news detection

Custom metrics, confusion matrix, error analysis

4-Model Export & Serving:

Save model & tokenizer (Hugging Face)

Export to ONNX

Predict using ONNX Runtime






🛠️ Requirements

1-Python 3.8+

2-pandas, scikit-learn, matplotlib, seaborn

3-nltk, gensim, emoji

4-torch, transformers, datasets, evaluate

5-onnx, onnxruntime

** Install requirements with:

pip install -r requirements.txt







🏃‍♂️ How to Run
1. Train & Fine-tune

Run the notebook fake_news_detection.ipynb (Colab, Kaggle, or Jupyter)

Follow the stages step by step, from EDA to deployment.

2. Serve or Export the Model

The fine-tuned model is saved to distilbert_fake_news_final/

ONNX model is exported as distilbert_fake_news.onnx








🙋‍♂️ Author

Ghassan Alkahlout
