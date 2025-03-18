import time

import numpy as np
import onnxruntime as ort
from transformers import MobileBertTokenizer

# Load the tokenizer
tokenizer = MobileBertTokenizer.from_pretrained("tokenizer_config.json")

# Load the ONNX model
ort_session = ort.InferenceSession("mobilebert_sentiment.onnx")

# Map indices back to emotion labels
id_to_emotion = {
    0: "UNRELATED",
    1: "NEUT",
    2: "POS",
    3: "NEG",
    4: "GIOIA",  # Joy
    5: "FIDUCIA",  # Trust
    6: "TRISTEZZA",  # Sadness
    7: "RABBIA",  # Anger
    8: "PAURA",  # Fear
    9: "DISGUSTO",  # Disgust
    10: "SORPRESA",  # Surprise
    11: "TREPIDAZIONE",  # Anticipation
    12: "SARCASMO",  # Sarcasm
    13: "AMORE",  # Love
    14: "COLPA",  # Guilt
    15: "CURIOSITA",  # Curiosity
    16: "ALLARME",  # Alarm
    17: "DISPERAZIONE",  # Despair
    18: "DELUSIONE",  # Disappointment
    19: "RIMORSO",  # Remorse
    20: "INVIDIA",  # Envy
    21: "CONTEMPO",  # Contempt
    22: "CINISMO",  # Cynicism
    23: "AGGRESSIONE",  # Aggression
    24: "ORGOGLIO",  # Pride
    25: "OTTIMISMO",  # Optimism
    26: "FATALISMO",  # Fatalism
    27: "DELIZIA",  # Delight
    28: "SENTIMENTALITA",  # Sentimentality
    29: "VERGOGNA",  # Shame
    30: "INDIGNAZIONE",  # Outrage
    31: "PESSIMISMO",  # Pessimism
    32: "MORBIDEZZA",  # Morbidness
    33: "DOMINANZA",  # Dominance
    34: "PARTECIPAZIONE",  # Participation
    35: "ANSIA",  # Anxiety
}


def classify_text(text):
    # Preprocess text
    inputs = tokenizer(
        text, return_tensors="np", truncation=True, padding="max_length", max_length=128
    )

    input_ids = np.array(inputs["input_ids"], dtype=np.int64)

    # Convert to NumPy arrays for ONNX Runtime
    ort_inputs = {"input": input_ids}

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
        if user_input.lower() == "exit":
            break
        classify_text(user_input)
