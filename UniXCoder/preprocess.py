import gc
import json
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Preprocess dataset and split into train, validation, and test sets.')
parser.add_argument('--datafile', type=str, default='classes-cleaned-dedup.jsonl', help='Path to the data file.')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Size ratio of the train set.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Size ratio of the validation set.')
parser.add_argument('--test_ratio', type=float, default=0.1, help='Size ratio of the test set.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling.')

# Parse arguments
args = parser.parse_args()

# Use the arguments
file_path = args.datafile
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio
random_seed = args.seed


def count_entries(file_path):
    with open(file_path, 'r') as file:
        for count, _ in enumerate(file, 1):
            pass
    return count

total_entries = count_entries(file_path)

def load_and_split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    # Load the data
    data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_entries):
        # data = [json.loads(line.strip()) for line in file]
        # data = [line for line in file]
            json_object = json.loads(line.strip())
            json_object = {'class': json_object['class'], 'docstring': json_object['docstring']}
            data.append(json_object)

    # Shuffle the data
    print(' Spliting the data')
    random.seed(random_seed)
    random.shuffle(data)
    # Split the data
    train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=random_seed)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(val_ratio+test_ratio), random_state=random_seed)
    
    return train_data, val_data, test_data

print('Loading and Splitting the data ...')
train_data, val_data, test_data = load_and_split_dataset(file_path, train_ratio, val_ratio, test_ratio, random_seed)
print(' Done')
gc.collect()  # Explicitly call garbage collector

print('Writing data to files ...')
def save_data_to_jsonl(data, file_name):
    print(f' Writing to {file_name}')
    with open(file_name, 'w') as file:
        for item in tqdm(data):
            json.dump(item, file)
            file.write('\n')

save_data_to_jsonl(train_data, 'train.jsonl')
save_data_to_jsonl(val_data, 'val.jsonl')
save_data_to_jsonl(test_data, 'test.jsonl')
save_data_to_jsonl(test_data[:100], 'experiment.jsonl')
print('Done')

# Delete the data variables
del train_data
del val_data
del test_data
gc.collect()  # Explicitly call garbage collector
