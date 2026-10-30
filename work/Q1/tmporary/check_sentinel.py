import os, glob
import pandas as pd

RES = r"C:\Users\ttylin\Documents\University\Big Data Competition\resources"
dirs = [d for d in glob.glob(os.path.join(RES, "*")) if os.path.isdir(d)]
for d in dirs:
    print("=" * 100)
    print("DIR:", os.path.basename(d))
    total_pos = total_neg = 0
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        df = pd.read_csv(f)
        num = df.select_dtypes(include="number")
        pos = (num == 9999.9).sum().sum()
        neg = (num == -9999.0).sum().sum()
        # near-sentinel values that are not exact
        near = num[(num > -10000.5) & (num < -9998.5)]
        total_pos += pos
        total_neg += neg
        cols_neg = {c: int((num[c] == -9999.0).sum()) for c in num.columns if (num[c] == -9999.0).any()}
        cols_pos = {c: int((num[c] == 9999.9).sum()) for c in num.columns if (num[c] == 9999.9).any()}
        print(f"{os.path.basename(f):28s} rows={len(df):6d} 9999.9={pos:6d} -9999.0={neg:5d}  min={num.min().min():12.4f}  max={num.max().max():12.4f}")
        if cols_neg:
            print("       NEG cols:", cols_neg)
        if cols_pos:
            print("       POS cols:", cols_pos)
        if len(near):
            print("       other values in (-10000.5,-9998.5):", len(near))
            print(near.head(10).to_string())
    print(f"TOTAL 9999.9={total_pos}  -9999.0={total_neg}")

# Show unique negative extremes across all files, to see if -9999-like sentinel variants exist
print("=" * 100)
print("Global min per column across all files (negative minima only):")
mins = {}
for f in glob.glob(os.path.join(RES, "**", "*.csv"), recursive=True):
    df = pd.read_csv(f)
    for c in df.select_dtypes(include="number").columns:
        m = df[c].min()
        if pd.notna(m) and m < -100:
            mins.setdefault(c, []).append((m, os.path.basename(f)))
for c, vals in sorted(mins.items(), key=lambda kv: min(v[0] for v in kv[1])):
    vals.sort()
    print(f"{c:45s} min={vals[0][0]:15.4f} ({vals[0][1]})")
