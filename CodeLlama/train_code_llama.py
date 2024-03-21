import argparse
from datetime import datetime
import sys
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

from torch.utils.data import DataLoader
from torch.optim import AdamW


class FineTuner:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None

    def load_datasets(self):
        train_dataset = load_dataset("json", data_files=self.args.train_filename)[
            "train"
        ]
        val_dataset = load_dataset("json", data_files=self.args.val_filename)["train"]

        # Since both datasets are loaded with the 'train' split, create a new DatasetDict to organize them
        datasets = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,  # Now, you manually set the validation split
            }
        )

        print("Train dataset:", datasets["train"])
        print("Val dataset:", datasets["validation"])

        return datasets["train"], datasets["validation"]

    def tokenize_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["class"],
            padding="max_length",
            # padding=False,
            truncation=True,
            max_length=self.args.max_source_length,
            return_tensors="pt",
        )
        print(examples["docstring"][:2])
        tokenized_outputs = self.tokenizer(
            examples["docstring"],
            padding="max_length",
            # padding=False,
            truncation=True,
            max_length=self.args.max_target_length,
            return_tensors="pt",
            # text_target=True,
        )
        tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
        print(len(tokenized_inputs["input_ids"]), len(tokenized_inputs["labels"]))
        return tokenized_inputs

    def tokenize_datasets(self, train_dataset, val_dataset):
        tokenized_train_dataset = train_dataset.map(
            self.tokenize_function, batched=True
        )
        tokenized_val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        return tokenized_train_dataset, tokenized_val_dataset

    def initialize_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # Set pad token to eos token

        # Prepare model for INT8 training and apply PEFT
        self.model.train()
        self.model = prepare_model_for_int8_training(self.model)

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        # Disable caching for PEFT model
        self.model.config.use_cache = False
        old_state_dict = self.model.state_dict
        self.model.state_dict = lambda *_, **__: get_peft_model_state_dict(
            self.model, old_state_dict()
        ).__get__(self.model, type(self.model))

    def setup_training(self, tokenized_train_dataset, tokenized_val_dataset):
        training_args = TrainingArguments(
            per_device_train_batch_size=self.args.train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            warmup_steps=100,
            learning_rate=self.args.learning_rate,
            logging_dir="./logs",
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            fp16=True,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=data_collator,
        )

        return trainer

    def train(self):
        train_dataset, val_dataset = self.load_datasets()
        self.initialize_model_and_tokenizer()
        tokenized_train_dataset, tokenized_val_dataset = self.tokenize_datasets(
            train_dataset, val_dataset
        )
        trainer = self.setup_training(tokenized_train_dataset, tokenized_val_dataset)
        trainer.train()

    def debug_train(self):
        # Load and prepare datasets
        train_dataset, val_dataset = self.load_datasets()
        self.initialize_model_and_tokenizer()
        tokenized_train_dataset, tokenized_val_dataset = self.tokenize_datasets(
            train_dataset, val_dataset
        )

        # Convert datasets to PyTorch DataLoader
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model, return_tensors="pt"
        )
        train_dataloader = DataLoader(
            tokenized_train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        val_dataloader = DataLoader(
            tokenized_val_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=data_collator,
        )

        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # Move model to the appropriate device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # print dataloader
        print(f"Train dataloader: {train_dataloader}")

        # print the first batch
        print(f"First batch: {train_dataloader[0]}")

        # Training loop
        # self.model.train()
        # for batch in train_dataloader:
        #     print(f"Batch {batch}")
        #     optimizer.zero_grad()

        #     input_ids = batch["input_ids"].to(device)
        #     attention_mask = batch["attention_mask"].to(device)
        #     labels = batch["labels"].to(device)

        #     outputs = self.model(
        #         input_ids=input_ids, attention_mask=attention_mask, labels=labels
        #     )

        #     # Debugging: Print shapes to understand the mismatch
        #     print(f"Input IDs shape: {input_ids.shape}")
        #     print(f"Attention Mask shape: {attention_mask.shape}")
        #     print(f"Labels shape: {labels.shape}")
        #     print(f"Outputs shape: {outputs.logits.shape}")

        #     loss = outputs.loss
        #     print(f"Loss: {loss}")

        #     loss.backward()
        #     optimizer.step()

        #     break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the validation set.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        required=True,
        help="The input training data file (a jsonl file).",
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        required=True,
        help="The input validation data file (a jsonl file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=768,
        help="The maximum total source sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total target sequence length after tokenization.",
    )
    parser.add_argument(
        "--beam_size", type=int, default=1, help="Beam size for searching."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fine_tuner = FineTuner(args)
    if args.do_train:
        # fine_tuner.train()
        fine_tuner.debug_train()


if __name__ == "__main__":
    main()
