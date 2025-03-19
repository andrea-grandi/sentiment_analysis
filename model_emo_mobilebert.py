import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

MODEL_NAME = "lordtt13/emo-mobilebert"
# MODEL_NAME = "JuliusAlphonso/distilbert-plutchik"
DATASET_PATH = "/kaggle/input/dataset-sa/MultiEmotions-It.tsv"
TEXT_COLUMN = "comment"
LABEL_COLUMN = "EMOTIONS"
OUTPUT_DIR = "/kaggle/working/"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
SEED = 42

# Reproducibility
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo in uso: {device}")


# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(file_path, text_col, label_col):
    """Carica il dataset da un file TSV."""
    df = pd.read_csv(file_path, sep="\t")

    # Verifica la presenza delle colonne necessarie
    if text_col not in df.columns or label_col not in df.columns:
        available_cols = ", ".join(df.columns)
        raise ValueError(
            f"Colonne richieste non trovate. Colonne disponibili: {available_cols}"
        )

    # Se le etichette sono testuali, convertiamole in numeriche
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        label_map = {label: idx for idx, label in enumerate(df[label_col].unique())}
        df["label_id"] = df[label_col].map(label_map)
        print(f"Mappatura etichette: {label_map}")
        return df[text_col].values, df["label_id"].values, label_map

    return df[text_col].values, df[label_col].values, None


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"labels shape: {labels.shape}")

        outputs = model(input_ids, attention_mask, labels)

        # print(outputs.logits.shape)
        # print(labels.shape)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1


def main():
    patience = 2
    patience_counter = 0

    # Crea la directory di output se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carica il dataset
    print(f"Caricamento del dataset da {DATASET_PATH}...")
    texts, labels, label_map = load_data(DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN)

    # Divisione in training e validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )

    print(f"Testi di training: {len(train_texts)}")
    print(f"Testi di validazione: {len(val_texts)}")

    # Numero di etichette uniche nel dataset
    num_labels = len(np.unique(labels))
    print(f"Numero di etichette: {num_labels}")

    # Carica il tokenizer e il modello
    print(f"Caricamento del modello {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.classifier = nn.Linear(model.config.hidden_size, num_labels)

    """
    for param in model.parameters():
        param.requires_grad = False

    # Sblocca i parametri del classificatore
    for param in model.classifier.parameters():
        param.requires_grad = True
    """

    # print(model)
    # print(f"PARAMETERS: {sum(p.numel() for p in model.parameters())}")

    # Prepara i dataset
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Prepara i dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Prepara l'ottimizzatore e lo scheduler
    optimizer = AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Sposta il modello sul dispositivo appropriato
    model.to(device)

    # Addestramento
    print("Inizio dell'addestramento...")
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoca {epoch + 1}/{EPOCHS}")

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )

        print(
            f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}"
        )

        val_loss, val_acc, val_f1 = evaluate(model, val_dataloader, device)

        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Salva il modello se abbiamo ottenuto un miglior F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patient_counter = 0

            # Salva il modello
            output_path = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Salva la mappatura delle etichette se presente
            if label_map:
                label_map_file = os.path.join(output_path, "label_map.txt")
                with open(label_map_file, "w") as f:
                    for label, idx in label_map.items():
                        f.write(f"{label}\t{idx}\n")

            print(f"Modello salvato in {output_path}")

        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                "Early stopping attivato: nessun miglioramento per troppe epoche consecutive."
            )
            break

    print("\nAddestramento completato!")
    print(f"Miglior F1 score di validazione: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
