# train_sentiment.py

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch

# 1. Load Dataset
raw = load_dataset("yelp_polarity")

# 2. Subset the training and test sets for faster training
train_data = raw["train"].shuffle(seed=42).select(range(200))
test_data = raw["test"].shuffle(seed=42).select(range(100))
dataset = {
    "train": train_data,
    "test": test_data
}

# 3. Preprocess
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized = {
    "train": dataset["train"].map(tokenize, batched=True),
    "test": dataset["test"].map(tokenize, batched=True)
}

for split in ["train", "test"]:
    tokenized[split] = tokenized[split].remove_columns(["text"])
    tokenized[split] = tokenized[split].rename_column("label", "labels")
    tokenized[split].set_format("torch")

# 4. Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 6. Train
trainer.train()

# 7. Save model
model.save_pretrained("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")

print("✅ Model fine-tuned and saved to ./sentiment-model")
