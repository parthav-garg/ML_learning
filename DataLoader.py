import numpy as np
import math

class DataLoader:
    """
    A simple and robust DataLoader for iterating over datasets in batches.
    Designed to work seamlessly with Python's `for ... in ...` loops.
    """
    def __init__(self, data, labels, batch_size=64, shuffle=True):
        """
        Initializes the DataLoader.

        Args:
            data (np.ndarray): The feature data.
            labels (np.ndarray): The corresponding labels.
            batch_size (int): The number of samples per batch.
            shuffle (bool): If True, shuffles the data at the start of each epoch.
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        
        # This will be used to track our position in the current epoch
        self.batch_index = 0

    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches

    def __iter__(self):
        """
        Resets the state for a new epoch. This is called automatically
        at the beginning of a `for` loop.
        """
        # Reset the batch counter for the new epoch
        self.batch_index = 0
        
        # Create and (optionally) shuffle the indices for this epoch
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        return self

    def __next__(self):
        """Returns the next batch of data. This is called automatically by a `for` loop."""
        # Check if we have yielded all batches for this epoch
        if self.batch_index >= self.num_batches:
            raise StopIteration

        # Calculate the start and end indices for the current batch
        start_idx = self.batch_index * self.batch_size
        end_idx = start_idx + self.batch_size
        
        # Get the indices for this specific batch from our shuffled list
        batch_indices = self.indices[start_idx:end_idx]

        # Use the batch indices to retrieve the data and labels
        x_batch = self.data[batch_indices]
        y_batch = self.labels[batch_indices]

        # Move to the next batch
        self.batch_index += 1

        return x_batch, y_batch