import os
import pickle
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import gc

class DynamicDataset(Dataset):
    def __init__(self, output_dir, base_filename, chunk_size=5000, cache_size=3):
        self.output_dir = output_dir
        self.base_filename = base_filename
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.chunk_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith('.pkl')]
        self.chunk_files.sort()  # Ensure chunks are loaded in order
        self.chunks_cache = OrderedDict()
        self.current_chunk = []
        self.current_chunk_index = -1
        self._load_chunk(0)

    def _load_chunk(self, index):
        if index in self.chunks_cache:
            self.current_chunk = self.chunks_cache.pop(index)
        else:
            print('readina new chunk:', index)
            with open(os.path.join(self.output_dir, self.chunk_files[index]), 'rb') as f:
                self.current_chunk = pickle.load(f)
            if len(self.chunks_cache) >= self.cache_size:
                self.chunks_cache.popitem(last=False)
                gc.collect()
        self.chunks_cache[index] = self.current_chunk
        self.current_chunk_index = index
        return self.current_chunk

    def __len__(self):
        return len(self.chunk_files) * self.chunk_size

    def __getitem__(self, idx):
        chunk_index = idx // self.chunk_size
        if chunk_index != self.current_chunk_index:
            self._load_chunk(chunk_index)
        item_index = idx % self.chunk_size
        feature = self.current_chunk[item_index]
        return torch.tensor(feature.source_ids, dtype=torch.long), torch.tensor(feature.target_ids, dtype=torch.long)
