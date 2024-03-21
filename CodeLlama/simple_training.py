import argparse
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


class SimpleFineTuner:
    def __init__(
        self,
        model_name_or_path,
        train_filename,
        val_filename,
        output_dir,
        max_length=256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.train_dataset = load_dataset("json", data_files=train_filename)["train"]
        self.val_dataset = load_dataset("json", data_files=val_filename)["train"]
        self.output_dir = output_dir
        self.max_length = max_length

        # Update tokenizer's padding token if necessary
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_and_encode(self, examples):
        # Tokenize the inputs and labels
        tokenized_inputs = self.tokenizer(
            examples["class"],
            truncation=True,
            # padding="longest",
            padding=False,
            max_length=10,
        )
        tokenized_labels = self.tokenizer(
            examples["docstring"],
            truncation=True,
            # padding="longest"
            padding=False,
            max_length=10,
        )
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    def train(self, batch_size=1, epochs=3):
        # Preparing the data collator

        # Move model to the appropriate device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("loading model to device:", device)
        # self.model.to(device)
        print("loading tokenizer to device:", device)
        # self.tokenizer.to(device)

        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        print("Tokenizing datasets...")

        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_and_encode, batched=True
        )

        print("Train dataset:", self.train_dataset)
        self.val_dataset = self.val_dataset.map(self.tokenize_and_encode, batched=True)

        print("Val dataset:", self.val_dataset)

        # Creating DataLoaders
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, collate_fn=data_collator
        )

        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Training loop
        self.model.train()
        for batch in train_dataloader:
            print(f"Batch {batch}")
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Debugging: Print shapes to understand the mismatch
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention Mask shape: {attention_mask.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Outputs shape: {outputs.logits.shape}")

            loss = outputs.loss
            print(f"Loss: {loss}")

            loss.backward()
            optimizer.step()

            break

        # # Trainer setup
        # training_args = TrainingArguments(
        #     output_dir=self.output_dir,
        #     num_train_epochs=epochs,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     logging_dir=f"{self.output_dir}/logs",
        #     logging_steps=10,
        #     evaluation_strategy="epoch",
        #     save_strategy="epoch",
        # )

        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=self.train_dataset,
        #     eval_dataset=self.val_dataset,
        #     data_collator=data_collator,
        #     tokenizer=self.tokenizer,
        # )

        # # Train
        # trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--val_filename", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fine_tuner = SimpleFineTuner(
        args.model_name_or_path, args.train_filename, args.val_filename, args.output_dir
    )
    fine_tuner.train()
