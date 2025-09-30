import gradio as gr
import onnxruntime as ort
from transformers import DistilBertTokenizerFast
import numpy as np

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_fake_news_final")

# Load ONNX model
onnx_path = "distilbert_fake_news.onnx"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def predict(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    logits = sess.run(None, ort_inputs)[0]
    pred = logits.argmax(axis=1).item()
    return "✅ True News" if pred == 1 else "❌ Fake News"

demo = gr.Interface(fn=predict, inputs="text", outputs="label", title="Fake Financial News Detector")

if __name__ == "__main__":
    demo.launch()
