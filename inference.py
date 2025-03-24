import time

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
model = AutoModelForSequenceClassification.from_pretrained("./models/")

# Map indices back to emotion labels
id_to_emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def classify_text(text):
    # Preprocess text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    )
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    
    # Get predictions
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_emotion = id_to_emotion[predicted_class]
    
    print(f"Text: {text}")
    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    
    return predicted_emotion

if __name__ == "__main__":
    sample_text = "Ti odio!!"
    classify_text(sample_text)

    # Interactive mode
    print("\\nEnter text to classify (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        classify_text(user_input)
