#!/usr/bin/env python3
"""Create train.list / val.list / test.list from ROOT files."""
from __future__ import annotations
import argparse, glob, random
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("pattern",help="Glob pattern for ROOT files, quote it in shell")
    p.add_argument("-o","--output-dir",default="lists")
    p.add_argument("--n-train-files",type=int,required=True); p.add_argument("--n-val-files",type=int,required=True); p.add_argument("--n-test-files",type=int,required=True)
    p.add_argument("--shuffle",action="store_true",help="Shuffle the sorted file list before splitting")
    p.add_argument("--seed",type=int,default=12345,help="Random seed used with --shuffle")
    return p.parse_args()

def write_list(path,files):
    with open(path,"w",encoding="utf-8") as f:
        for name in files: f.write(str(Path(name).resolve())+"\n")
    print(f"[write] {path} n_files={len(files)}")

def main():
    a=parse_args(); files=sorted(glob.glob(a.pattern))
    if not files: raise SystemExit(f"No files match pattern: {a.pattern}")
    if a.shuffle:
        rng=random.Random(a.seed); rng.shuffle(files); print(f"[split] shuffled file order with seed={a.seed}")
    n=a.n_train_files+a.n_val_files+a.n_test_files
    if len(files)<n: raise SystemExit(f"Need at least {n} files, found {len(files)}")
    tr=files[:a.n_train_files]; va=files[a.n_train_files:a.n_train_files+a.n_val_files]; te=files[a.n_train_files+a.n_val_files:n]
    d=Path(a.output_dir); d.mkdir(parents=True,exist_ok=True); write_list(d/"train.list",tr); write_list(d/"val.list",va); write_list(d/"test.list",te)

if __name__=="__main__": main()
