import os, glob
import pandas as pd
import numpy as np

RES = r"C:\Users\ttylin\Documents\University\Big Data Competition\resources"
EN = os.path.join(RES, "1. Hourly Comprehensive Observation Dataset of Land-Atmosphere Interaction in the High-Cold Mountainous Region of Jilong, Xizang (2021\u20132026)")

print("EN dir exists:", os.path.isdir(EN))
print("build_panel files:", len(glob.glob(os.path.join(EN, "*_Gyirong_*.csv"))))

for d in [EN] + [x for x in glob.glob(os.path.join(RES, "*")) if os.path.isdir(x) and x != EN]:
    print("=" * 90)
    print("DIR:", os.path.basename(d))
    tot = 0
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        df = pd.read_csv(f)
        num = df.select_dtypes(include="number")
        n_neg = int((num == -9999.0).sum().sum())
        n_pos = int((num == 9999.9).sum().sum())
        # true element-wise near-sentinel count
        arr = num.to_numpy(dtype=float)
        near_mask = np.isfinite(arr) & (arr > -10000.5) & (arr < -9998.5)
        near_vals = np.unique(arr[near_mask]) if near_mask.any() else []
        n_int_neg = int((num == -9999).sum().sum())
        tot += n_neg
        print(f"  {os.path.basename(f):30s} -9999.0={n_neg:4d} -9999(int)={n_int_neg:4d} 9999.9={n_pos:6d} near_neg={int(near_mask.sum()):4d} {near_vals[:5]}")
    print("  SUBTOTAL exact -9999.0:", tot)

print("=" * 90)
print("Any string cell exactly '-9999' / '-9999.0' anywhere in resources:")
hits = 0
for f in glob.glob(os.path.join(RES, "**", "*.csv"), recursive=True):
    df = pd.read_csv(f, dtype=str)
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        m = s.isin(["-9999", "-9999.0", "-9999.00"])
        if m.any():
            hits += 1
            print(f, c, int(m.sum()), s[m].unique()[:5])
print("files/columns with string -9999:", hits)
