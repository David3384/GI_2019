import numpy as np
import collections

# removing the padded 0s' at the end while keeping the length at least minimum_length
def trim(padded, minimum_length=0):
    trimmed = [x for x in padded if x != 0]
    if len(trimmed) < minimum_length:
        for i in range(minimum_length - len(trimmed)):
            trimmed.append(0)
    return trimmed

def pad_with_np_zeros(arr, max_len=50):
    if len(arr) > 50:
        return arr[:50]
    else:
        pad_val = np.zeros(arr[0].shape)
        arr = [a for a in arr] + [pad_val] * 50
        arr = arr[:50]
    return np.array(arr)

def add_dicts(d1, d2):
    result = collections.defaultdict(int)
    for key in d1:
        result[key] += d1[key]
    for key in d2:
        result[key] += d2[key]
    return result

def avg_dict(d):
    return sum([d[key] * key for key in d]) / sum([d[key] for key in d])