# benchmarks/synth_data.py
import numpy as np
import os

def make_synthetic(n=100000, d=128, out='data/raw.npy'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X = np.random.normal(size=(n, d)).astype('float32')
    np.save(out, X)
    print('Saved', out)

if __name__ == '__main__':
    make_synthetic(20000, 128, out='data/raw.npy')