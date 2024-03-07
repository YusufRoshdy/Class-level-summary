from datetime import datetime
import os
import sys
import torch
from datasets import load_dataset
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
    set_peft_model_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
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
        "--dev_filename",
        type=str,
        required=True,
        help="The input development data file (a jsonl file).",
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
        default=10,
        help="Total number of training epochs to perform.",
    )

    args = parser.parse_args()
    return args


# Load custom dataset
train_dataset = load_dataset("json", data_files="train.jsonl", split="train")
val_dataset = load_dataset("json", data_files="val.jsonl", split="train")

# Initialize model and tokenizer
base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Ensure correct tokenizer configuration
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    class_code = data_point["class"]
    docstring = data_point["docstring"]
    full_prompt = f"""### Python class code:
{class_code}

### Docstring:
{docstring}
"""
    return tokenize(full_prompt)


# Tokenize datasets
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt, batched=True)
tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt, batched=True)

# Prepare model for INT8 training and apply PEFT
model.train()
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Configuration for training
batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "code-classification"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    max_steps=400,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=20,
    save_steps=20,
    output_dir=output_dir,
    group_by_length=True,
    report_to="wandb",
    run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

# Disable caching for PEFT model and prepare for training
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = lambda self, *_, **__: get_peft_model_state_dict(
    self, old_state_dict()
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

# Start training
trainer.train()
