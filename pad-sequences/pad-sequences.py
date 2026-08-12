import numpy as np
import math
def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        lens = [len(i) for i in seqs]
        print(lens)
        max_len = max(lens)
        print(max_len)
    padded = []
    for i in range(len(seqs)): 
        print(seqs[i])
        values_to_add = max(0, max_len - len(seqs[i]))
        values_to_truncate = max(0, len(seqs[i]) - max_len)
        print(values_to_add, values_to_truncate)
        if values_to_add > 0:
            updated = seqs[i] + [pad_value] * values_to_add
        elif values_to_truncate > 0: 
            updated = seqs[i][:-values_to_truncate]
        else: 
            updated = seqs[i]

        padded.append(updated)
        print(padded)
    return padded