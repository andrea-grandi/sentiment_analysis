import torch
import time
from transformers import MobileBertTokenizer
import onnxruntime as ort
import numpy as np

# Load the tokenizer
tokenizer = MobileBertTokenizer.from_pretrained("./model")

# Load the ONNX model
ort_session = ort.InferenceSession("./model/mobilebert_sentiment.onnx")

# Map indices back to emotion labels
id_to_emotion = {0: "LOVE", 1: "TRUST", 2: "SADNESS", ...}  # Fill in with your actual mapping

def classify_text(text):
    # Preprocess text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Convert to NumPy arrays for ONNX Runtime
    ort_inputs = {
        'input': inputs['input_ids'].numpy()
    }
    
    # Run inference
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    # Get predictions
    logits = ort_outputs[0]
    predicted_class = np.argmax(logits, axis=1)[0]
    predicted_emotion = id_to_emotion[predicted_class]
    
    print(f"Text: {text}")
    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    
    return predicted_emotion

# Example usage
if __name__ == "__main__":
    sample_text = "I'm really happy with the performance of this model!"
    classify_text(sample_text)
    
    # Interactive mode
    print("\\nEnter text to classify (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        classify_text(user_input)
