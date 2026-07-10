"""
Script for cloudsen12 dataset extraction
Usage:
>>> python3 download.py
"""



import os
import sys
import csv
import tacoreader
import rasterio as rio



# B02=2, B03=3, B04=4, B08=8, B10=11, B11=12, B12=13
BANDS = [2, 3, 4, 8, 11, 12, 13]
OUT = sys.argv[1] if len(sys.argv) > 1 else "data"

for s in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT, s), exist_ok=True)

df = tacoreader.load("tacofoundation:cloudsen12-l1c")
high = df[df["label_type"] == "high"].reset_index(drop=True)
print(f"High-patches processed: {len(high)}")

meta = open(os.path.join(OUT, "metadata.csv"), "w", newline="", encoding="utf-8")
writer = csv.writer(meta)
writer.writerow(["index", "split", "roi_id", "s2_id", "img", "label"])

total = 0
for i in range(len(high)):
    row = high.iloc[i]
    split = str(row["tortilla:data_split"])
    if split == "validation":
        split = "val"
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    img_path = f"{split}/{i:06d}_img.tif"
    lab_path = f"{split}/{i:06d}_label.tif"

    if os.path.exists(os.path.join(OUT, img_path)) and \
       os.path.exists(os.path.join(OUT, lab_path)):
        writer.writerow([i, split, row["roi_id"], row["s2_id"], img_path, lab_path])
        total += os.path.getsize(os.path.join(OUT, img_path))
        continue

    sample = high.read(i)
    with rio.open(sample.read(0)) as src:
        arr = src.read(BANDS)          # (7, H, W) data={B02, B03, B04, B08, B10, B11, B12}
        prof = src.profile
    with rio.open(sample.read(1)) as src:
        mask = src.read(1)             # (H, W) labels={0, 1, 2, 3}
        mprof = src.profile

    for k in ("blockxsize", "blockysize", "tiled", "interleave"):
        prof.pop(k, None)
        mprof.pop(k, None)
    prof.update(count=7, compress="deflate", predictor=2, tiled=False)
    with rio.open(os.path.join(OUT, img_path), "w", **prof) as dst:
        dst.write(arr)
    mprof.update(count=1, compress="deflate", tiled=False)
    with rio.open(os.path.join(OUT, lab_path), "w", **mprof) as dst:
        dst.write(mask, 1)

    writer.writerow([i, split, row["roi_id"], row["s2_id"], img_path, lab_path])
    total += os.path.getsize(os.path.join(OUT, img_path))

    if i % 50 == 0:
        print(f"{i}/{len(high)}  {total/1e9:.2f} GB")
        meta.flush()

meta.close()
print(f"Completed. {total/1e9:.2f} GB in dir '{OUT}'.")
