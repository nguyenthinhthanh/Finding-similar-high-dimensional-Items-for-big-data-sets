# benchmarks/synth_data.py
import os
import numpy as np
import matplotlib.pyplot as plt

def make_synthetic(n=100000, d=128, out='data/raw.npy'):
    """Generate synthetic data (n samples, d dimensions) and save to a .npy file."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X = np.random.normal(size=(n, d)).astype('float32')
    np.save(out, X)
    print(f"Saved synthetic data to {out}")
    return X

def inspect_data(X):
    """Display basic statistics and sample rows of the dataset."""
    print("\nData inspection:")
    print(f"- Shape: {X.shape}")
    print(f"- Mean: {np.mean(X):.4f}")
    print(f"- Std deviation: {np.std(X):.4f}")
    print(f"- Min value: {np.min(X):.4f}")
    print(f"- Max value: {np.max(X):.4f}")
    
    print("\nSample rows:")
    print(X[:5])

if __name__ == '__main__':
    X = make_synthetic(20000, 128, out='data/raw.npy')
    inspect_data(X)
