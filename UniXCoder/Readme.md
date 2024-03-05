# UniXcoder

## Preprocessing
To preprocess the data and split it to train, val and test run the following command
```sh
python preprocess.py --datafile ../classes-cleaned-dedup.jsonl
```

This cmmand will also creat `expirament.jsonl` with 100 sampels to be used for debugging

## Fine-Tune Setting
Adjust the paramenters based on your resources

```sh
# Training
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename train.jsonl \
	--dev_filename  val.jsonl \
	--output_dir saved_models \
	--max_source_length 768 \
	--max_target_length 256 \
	--beam_size 1 \
	--train_batch_size 8 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10

# Evaluating	
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename test.jsonl \
	--output_dir saved_models \
	--max_source_length 768 \
	--max_target_length 256 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 	
```

to make sure that everything is working propererly, you can try to run with the `experiment.jsonl` first
```sh
# experiment
python run.py \
	--do_train \
    --do_eval \
	--model_name_or_path microsoft/unixcoder-base-nine \
	--train_filename experiment.jsonl \
	--dev_filename  experiment.jsonl \
	--output_dir saved_models \
	--max_source_length 768 \
	--max_target_length 256 \
	--beam_size 1 \
	--train_batch_size 8 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 1
```
