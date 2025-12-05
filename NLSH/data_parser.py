```


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse



def read_idx(filename):
    with open(filename, 'rb') as f:
        # Read the magic number (4 bytes)
        magic = int.from_bytes(f.read(4), byteorder='big')
        dtype_code = (magic >> 8) & 0xFF        # data type
        dims = magic & 0xFF                     # number of dimensions
        
        # Read the dimension sizes
        shape = tuple(int.from_bytes(f.read(4), 'big') for _ in range(dims))
        
        # Read the actual data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        
        return data


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    
    d = data[0]  # all vectors have same dimension
    if d <= 0:
        raise ValueError("Invalid vector dimension: {}".format(d))
    
    n = data.size // (d + 1)
    data = data.reshape(n, d + 1)
    vectors = data[:, 1:].view(np.float32)
    
    return vectors
    


def get_dataset(DataType, file_path):
    if DataType == "mnist":
        data = read_idx(file_path).astype(np.float32) / 255.0
        return data.reshape(data.shape[0], -1)

    elif DataType == "sift":
        data = read_fvecs(file_path)
        return data

    else:
        print("Insanity check - Dataset\n")
        print("\t No such type")





def search_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-q", "--query", type=str)
    # parser.add_argument("-i", "--index")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-type", "--type", default="mnist")
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-R", type=int)
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-range", type=bool, default=False)
    args = parser.parse_args()
    
    # Set default for -R
    if args.R is None:
        if args.type == "mnist":
            args.R = 2000
        elif args.type == "sift":
            args.R = 2800

    return args

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-q", "--query", type=str)
    # parser.add_argument("-i", "--index")
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("-type", "--type", default="mnist")
    parser.add_argument("-knn","--knn", type=int, default=10)
    parser.add_argument("-m","--m", type=int, default=100)
    parser.add_argument("-imbalance","--imbalance", type=float, default=0.03)
    parser.add_argument("-kahip_mode","--kahip_mode", type=int, default=2)
    parser.add_argument("-layers","--layers", type=int, default=3)
    parser.add_argument("-nodes","--nodes", type=int, default=64)
    parser.add_argument("-epochs","--epochs", type=int, default=10)
    parser.add_argument("-batch_size","--batch_size", type=int, default=128)
    parser.add_argument("-lr","--lr", type=float, default=0.001)
    parser.add_argument("-seed","--seed", type=int, default=1)
    args = parser.parse_args()

    return args
```