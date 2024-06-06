# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
from time import time
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)

import bleu
from DynamicDataset import DynamicDataset

logging.basicConfig(filename='log.txt', format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, idx, code, target, nl=None):
        self.example_id = idx
        self.source_ids = code
        self.target_ids = target 
        self.nl = nl 

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        



def train(args, model, device, tokenizer, writer):
    # Prepare training data loader
    tokenized_train_dir = args.train_filename.replace('.jsonl', '')
    if not os.path.exists(tokenized_train_dir):
        raise FileNotFoundError(f"Tokenized training data not found in {tokenized_train_dir}")

    train_dataset = DynamicDataset(tokenized_train_dir, 'chunk')
    chunk_files = train_dataset.chunk_files
    num_chunks = len(chunk_files)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(len(train_dataset) * args.num_train_epochs * 0.1 / args.train_batch_size),
                                                num_training_steps=len(train_dataset) * args.num_train_epochs / args.train_batch_size)

    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Num epoch = %d", args.num_train_epochs)

    model.train()
    patience, best_bleu, losses, dev_dataset = 0, 0, [], {}
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0  # For tracking epoch loss
        
        # Shuffle the chunk order for this epoch
        chunk_order = list(range(num_chunks))
        random.shuffle(chunk_order)

        for i, chunk_idx in enumerate(chunk_order):
            print(f'\t {i}- Training on chunk_{chunk_idx}')
            train_dataset._load_chunk(chunk_idx)
            train_sampler = SequentialSampler(train_dataset.current_chunk)
            train_dataloader = DataLoader(train_dataset.current_chunk, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

            # for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            for idx in tqdm(range(0, len(train_dataset.current_chunk), args.train_batch_size)):
                features = train_dataset.current_chunk[idx: idx + args.train_batch_size]
                source_ids = torch.tensor([feature.source_ids for feature in features], dtype=torch.long).to(device)
                target_ids = torch.tensor([feature.target_ids for feature in features], dtype=torch.long).to(device)
                st = time()
                # batch = tuple(t.to(device) for t in batch)
                # source_ids, target_ids = batch
                model_st = time()
                loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)
                # print('Model time', time() - model_st, 'For', len(batch))
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    epoch_loss += loss.item()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                                      len(losses) // args.gradient_accumulation_steps,
                                                                      round(np.mean(losses[-100 * args.gradient_accumulation_steps:]), 4)))
                # print('batch time', time() - st)

        # Log epoch loss to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss / len(losses) // args.gradient_accumulation_steps), epoch)

        if args.do_eval:
            evaluate(args, model, device, tokenizer, writer, epoch, dev_dataset, best_bleu, patience)


def evaluate(args, model, device, tokenizer, writer, epoch, dev_dataset, best_bleu, patience):
    # Eval model with dev dataset
    tokenized_dev_dir = args.dev_filename.replace('.jsonl', '')
    if not os.path.exists(tokenized_dev_dir):
        raise FileNotFoundError(f"Tokenized dev data not found in {tokenized_dev_dir}")

    eval_dataset = DynamicDataset(tokenized_dev_dir, 'chunk')
    chunk_files = eval_dataset.chunk_files
    num_chunks = len(chunk_files)

    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Start Evaling model
    model.eval()
    eval_loss, tokens_num = 0, 0

    saved_features = []

    for i, chunk_idx in enumerate(range(num_chunks)):
        logger.info(f'\t {i}- Evaluating on chunk_{chunk_idx}')
        eval_dataset._load_chunk(chunk_idx)
        
        for idx in tqdm(range(0, len(eval_dataset.current_chunk), args.eval_batch_size)):
            features = eval_dataset.current_chunk[idx: idx + args.eval_batch_size]
            source_ids = torch.tensor([feature.source_ids for feature in features], dtype=torch.long).to(device)
            target_ids = torch.tensor([feature.target_ids for feature in features], dtype=torch.long).to(device)

            with torch.no_grad():
                _, loss, num = model(source_ids=source_ids, target_ids=target_ids)
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()

            if len(saved_features) < 1000:
                saved_features.extend(features)
        break

    eval_loss = eval_loss / tokens_num
    result = {'eval_ppl': round(np.exp(eval_loss), 5)}
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    logger.info("  " + "*" * 20)

    # Log evaluation loss to TensorBoard
    writer.add_scalar('Loss/eval', eval_loss, epoch)

    # Calculate BLEU
    logger.info("***** Calculating BLEU Score *****")
    p = []
    eval_examples = []
    # for features in tqdm(saved_features, desc="Calculating BLEU"):
    for idx in tqdm(range(0, len(saved_features), args.eval_batch_size), desc="Calculating BLEU"):
        source_ids = torch.tensor([feature.source_ids for feature in saved_features[idx: idx+args.eval_batch_size]], dtype=torch.long).to(device)

        with torch.no_grad():
            preds = model(source_ids)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
        eval_examples.extend(features)

    model.train()
    predictions = []
    with open(args.output_dir + "/dev.output", 'w') as f, open(args.output_dir + "/dev.gold", 'w') as f1:
        for ref, gold in zip(p, eval_examples):
            predictions.append(str(gold.example_id) + '\t' + ref)
            f.write(str(gold.example_id) + '\t' + ref + '\n')
            f1.write(str(gold.example_id) + '\t' + gold.nl + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  " + "*" * 20)

    print("  %s = %s " % ("bleu-4", str(dev_bleu)))
    print("  " + "*" * 20)
    
    # Log BLEU score to TensorBoard
    writer.add_scalar('BLEU/dev', dev_bleu, epoch)

    if dev_bleu > best_bleu:
        logger.info("  Best bleu:%s", dev_bleu)
        logger.info("  " + "*" * 20)
        best_bleu = dev_bleu
        # Save best checkpoint for best bleu
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        output_model_file = os.path.join(output_dir, f"pytorch_model_{epoch}.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        patience = 0
    else:
        patience += 1
        if patience == 2:
            return best_bleu
    return best_bleu


def test(args, model, device, tokenizer, writer):
    # Load the best model
    checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(output_dir))

    # Prepare test data loader
    tokenized_test_dir = args.test_filename.replace('.jsonl', '')
    if not os.path.exists(tokenized_test_dir):
        raise FileNotFoundError(f"Tokenized test data not found in {tokenized_test_dir}")

    test_dataset = DynamicDataset(tokenized_test_dir, 'chunk')
    chunk_files = test_dataset.chunk_files
    num_chunks = len(chunk_files)

    logger.info("\n***** Running testing *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Start testing model
    model.eval()
    p = []
    test_examples = []

    for i, chunk_idx in enumerate(range(num_chunks)):
        logger.info(f'\t {i}- Testing on chunk_{chunk_idx}')
        test_dataset._load_chunk(chunk_idx)
        
        for idx in tqdm(range(0, len(test_dataset.current_chunk), args.eval_batch_size)):
            features = test_dataset.current_chunk[idx: idx + args.eval_batch_size]
            source_ids = torch.tensor([feature.source_ids for feature in features], dtype=torch.long).to(device)

            with torch.no_grad():
                preds = model(source_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
            test_examples.extend(features)

    model.train()
    predictions = []
    with open(args.output_dir + "/test.output", 'w') as f, open(args.output_dir + "/test.gold", 'w') as f1:
        for ref, gold in zip(p, test_examples):
            predictions.append(str(gold.example_id) + '\t' + ref)
            f.write(str(gold.example_id) + '\t' + ref + '\n')
            f1.write(str(gold.example_id) + '\t' + gold.nl + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test.gold"))
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    logger.info("  %s = %s " % ("bleu-4", str(test_bleu)))
    logger.info("  " + "*" * 20)

    print("  %s = %s " % ("bleu-4", str(test_bleu)))
    print("  " + "*" * 20)

    # Log test BLEU score to TensorBoard
    writer.add_scalar('BLEU/test', test_bleu, 0)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    # Initialize SummaryWriter
    writer = SummaryWriter()


    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 

    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)   
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args, model, device, tokenizer, writer)
                
if __name__ == "__main__":
    main()


