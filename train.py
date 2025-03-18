import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, MobileBertForSequenceClassification, MobileBertTokenizer

# from inference import *


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            # return_tensors="pt",
        )
        # print(encoding["attention_mask"])

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep="\t")

    # Extract the text for classification
    texts = df["comment"].tolist()

    # Extract the labels
    labels = df["EMOTIONS"].tolist()

    # Convert emotions in numerical labels
    unique_emotions = list(set(labels))
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    numeric_labels = [emotion_to_id[emotion] for emotion in labels]

    return texts, numeric_labels, emotion_to_id


def train_model(train_loader, val_loader, num_labels, epochs=1):
    # Load MobileBERT
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", num_labels=num_labels
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            print(attention_mask.shape)
            # print(input_ids.shape)
            # print(model)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        val_accuracy = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                preds = torch.argmax(outputs.logits, dim=1)
                val_accuracy += (preds == labels).sum().item()
                val_steps += len(labels)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader)}")
        print(f"Validation Loss: {val_accuracy / val_steps}")

        return model


def optimize_for_raspberry_pi(model, tokenizer, output_dir):
    # Convert to quantized model to reduce size and improve inference speed
    # Use torch.quantization for 8-bit quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(
        quantized_model.state_dict(), f"{output_dir}/mobilebert_sentiment_quantized.pt"
    )

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    # Export to ONNX for better performance (optional)
    dummy_input = torch.randint(1, 10000, (1, 128)).to("cpu")
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_dir}/mobilebert_sentiment.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def main():
    # Set up paths
    data_path = "./dataset/MultiEmotions-It.tsv"
    output_dir = "./mobilebert_sentiment_model"

    # Set up tokenizer
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

    # Load and preprocess data
    texts, labels, emotion_to_id = load_and_preprocess_data(data_path)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = MyDataset(train_texts, train_labels, tokenizer)
    val_dataset = MyDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train the model
    num_labels = len(emotion_to_id)
    model = train_model(train_loader, val_loader, num_labels)

    # Optimize and save for Raspberry Pi
    optimize_for_raspberry_pi(model, tokenizer, output_dir)

    print(f"Model trained, optimized, and saved to {output_dir}")


if __name__ == "__main__":
    main()
