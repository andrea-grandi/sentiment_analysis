import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Load the ONNX model
ort_session = ort.InferenceSession("./models/mobilebert_sentiment.onnx")

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
        text, return_tensors="np", truncation=True, padding="max_length", max_length=128
    )

    input_ids = np.array(inputs["input_ids"], dtype=np.int64)
    # print(input_ids)

    # Convert to NumPy arrays for ONNX Runtime
    ort_inputs = {"input_ids": input_ids}

    # print(ort_inputs)
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
