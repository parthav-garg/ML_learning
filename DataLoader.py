import numpy as np
from base.tensor import Tensor
import math

class DataLoader:
    def __init__(self, data, labels, batch_size=64, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        start = self.current_index
        end = start + self.batch_size
        batch_indices = self.indices[start:end]

        x_batch = self.data[batch_indices]
        y_batch = self.labels[batch_indices]

        self.current_index += self.batch_size

        return x_batch, y_batch