import pickle
from tqdm import tqdm
import pickle
import os
import json
import argparse

from transformers import RobertaTokenizer

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, idx, code, target, nl):
        self.example_id = idx
        self.source_ids = code
        self.target_ids = target 
        self.nl = nl 

def tokenize_file(filename, tokenizer, args, stage=None, chunk_size=5000, total=1272829):
    """Read examples from filename and convert them to token ids"""
    
    output_dir = filename.replace('.jsonl', '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk = []
    features = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), total=total):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx 
            code = js['class'].replace('\n',' ')
            code = ' '.join(code.strip().split())
            nl = js['docstring'].replace('\n','')
            nl = ' '.join(nl.strip().split())        
            
            # convert_examples_to_features
            #source
            source_tokens = tokenizer.tokenize(code)[:args.max_source_length-5]
            source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
            padding_length = args.max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id]*padding_length
    
            #target
            if stage=="test":
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(nl)[:args.max_target_length-2]
            target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
        
            # features.append(
            #     InputFeatures(
            #         idx,
            #         source_ids,
            #         target_ids,
            #         nl,
            #     )
            # )

            chunk.append(InputFeatures(idx, source_ids, target_ids, nl))

            # chunk
            if (idx+1) % chunk_size == 0:
                chunk_file = os.path.join(output_dir, f'chunk_{idx // chunk_size}.pkl')
                with open(chunk_file, 'wb') as cf:
                    pickle.dump(chunk, cf)
                chunk = []
    return

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    ## Other parameters
    parser.add_argument("--max_source_length", default=768, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=256, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.train_filename:
        tokenize_file(args.train_filename, tokenizer, args, stage='train', total=1272829) # 1272829
    if args.dev_filename:
        tokenize_file(args.dev_filename, tokenizer, args, stage='dev', total=159104) # 159104
    if args.test_filename:
        tokenize_file(args.test_filename, tokenizer, args, stage='test', total=159104) # 159104

if __name__ == "__main__":
    main()