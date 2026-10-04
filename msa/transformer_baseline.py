"""OPTIONAL deep-learning baseline (fine-tune DistilBERT). Needs: pip install transformers datasets torch
   python -m msa.transformer_baseline --data "data/IMDB Dataset.csv" --epochs 1 --sample 10000
A GPU is strongly recommended. Written to compare against the classical models; not run in the packaged tests."""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data import load_any, split_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="outputs/transformer")
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

    tr, te = split_data(load_any(a.data), sample=a.sample)
    tok = AutoTokenizer.from_pretrained(a.model)

    def enc(df):
        ds = Dataset.from_dict({"text": df.review.tolist(), "label": (df.label == "positive").astype(int).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    args = TrainingArguments(a.out, num_train_epochs=a.epochs, per_device_train_batch_size=16,
                             per_device_eval_batch_size=32, learning_rate=2e-5, report_to=[], save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=enc(tr), eval_dataset=enc(te), tokenizer=tok)
    trainer.train()
    logits = trainer.predict(enc(te)).predictions
    y, p = (te.label == "positive").astype(int).to_numpy(), np.argmax(logits, axis=1)
    print(f"Transformer accuracy={accuracy_score(y, p):.4f} macro-F1={f1_score(y, p, average='macro'):.4f}")


if __name__ == "__main__":
    main()
