
#!/usr/bin/env python3
import argparse, numpy as np
from collections import Counter
from data import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--win', type=int, default=60)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--features', type=str, default='xyconf')
    ap.add_argument('--norm', type=str, default='image')
    args = ap.parse_args()

    X, y, classes = load_dataset(args.data_root, args.win, args.stride, args.features, args.norm)
    counts = Counter(y.tolist())
    print(f"X={X.shape}, classes={classes}")
    for i,c in enumerate(classes):
        print(f"  {i}:{c:>15s} -> {counts.get(i,0)} sequences")
    if len(classes) > 0:
        imbalance = max(counts.values())/max(1,min(counts.values()))
        print(f"class imbalance ratio (max/min): {imbalance:.2f}")

if __name__ == '__main__':
    main()
