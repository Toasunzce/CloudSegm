"""
Submodule for scene fetching from copernicus/eu client of Sentinel2 dataset.
Usage example:

# full param set
>>> rows = fetch_and_save_patches(
        lon=27.5615, lat=53.9045,
        start_date="2024-05-01",
        end_date="2024-08-30",
        location_name="minsk_summer24",
        max_scenes=8,
        min_cloud_pct=10,
        max_cloud_pct=60,
        path="examples",
    )
>>> save_metadata(rows, "minsk_summer24")

# simplified solution
>>> rows = fetch_and_save_patches(
        lon=0.2156,
        lat=52.3797,
        start_date="2024-05-01",
        end_date="2024-08-30",
    )
>>> save_metadata(rows)
"""



import os
import csv
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_bounds
import torch
from torch.utils.data import Dataset
from pystac_client import Client
import odc.stac



def fetch_and_save_patches(lon, lat,
                            start_date, end_date,
                            location_name="loc",
                            max_scenes=8,
                            min_cloud_pct=0,
                            max_cloud_pct=80,
                            path="out_data",):
    
    BANDS = ["B02", "B03", "B04", "B08", "B10", "B11", "B12"]
    PATCH_NATIVE = 509
    PATCH_PX     = 512
    RES_M        = 10

    location_dir = os.path.join(path, location_name)
    os.makedirs(location_dir, exist_ok=True)

    lat_delta = (PATCH_NATIVE * RES_M) / 111000 / 2
    lon_delta = lat_delta / np.cos(np.radians(lat))

    bbox = (
        lon - lon_delta, 
        lat - lat_delta, 
        lon + lon_delta, 
        lat + lat_delta
    )

    client  = Client.open("https://catalogue.dataspace.copernicus.eu/stac")
    results = client.search(
        collections=["sentinel-2-l1c"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_pct,
                "gt": min_cloud_pct,
            }
        },
        sortby=[{"field": "datetime", "direction": "asc"}],
        max_items=max_scenes,
    )
    items = list(results.items())
    print(f"Scenes found: {len(items)}")
    for item in items:
        assert item.datetime is not None
        print(f"  {item.datetime.date()}  "
              f"cloud={item.properties.get('eo:cloud_cover', '?')}%  "
              f"id={item.id}")

    if not items:
        return []

    def _sign_cdse(url):
        return url.replace("s3://eodata/", "/vsis3/eodata/")

    meta_rows = []

    for t_idx, item in enumerate(items):
        scene_id = item.id
        assert item.datetime is not None
        date_str = str(item.datetime.date())
        print(f"[{t_idx+1}/{len(items)}] Loading {date_str} {scene_id[:40]}...")

        try:
            ds = odc.stac.load(
                [item],
                bands=BANDS,
                bbox=bbox,
                resolution=RES_M,
                dtype="uint16",
                patch_url=_sign_cdse,
            )
        except Exception as e:
            print(f"  Fetching error: {e}")
            continue

        arr = np.stack([
            ds[b].isel(time=0).values for b in BANDS
        ])  # (7, H, W) uint16

        H, W = arr.shape[1], arr.shape[2]
        print(f"  Size: {H}x{W}")

        if H < PATCH_NATIVE or W < PATCH_NATIVE:
            print(f"  Skip: patch size is lower than {PATCH_NATIVE}")
            continue

        h0  = (H - PATCH_NATIVE) // 2
        w0  = (W - PATCH_NATIVE) // 2
        arr = arr[:, h0:h0+PATCH_NATIVE, w0:w0+PATCH_NATIVE]
        arr = np.pad(
            arr,
            pad_width=((0, 0),
                       (0, PATCH_PX - PATCH_NATIVE),
                       (0, PATCH_PX - PATCH_NATIVE)),
            mode="constant",
            constant_values=0,
        )  # (7, 512, 512)

        lon_min, lat_min, lon_max, lat_max = bbox
        res_deg_x     = (lon_max - lon_min) / W
        res_deg_y     = (lat_max - lat_min) / H
        patch_lon_min = lon_min + w0 * res_deg_x
        patch_lat_max = lat_max - h0 * res_deg_y
        patch_lon_max = patch_lon_min + PATCH_NATIVE * res_deg_x
        patch_lat_min = patch_lat_max - PATCH_NATIVE * res_deg_y

        transform = from_bounds(
            patch_lon_min, patch_lat_min,
            patch_lon_max, patch_lat_max,
            PATCH_PX, PATCH_PX,
        )
        tile = scene_id.split("_")[-2]
        fname = f"{location_name}_{date_str}_{tile}"
        img_name = f"{fname}_img.tif"
        img_path = os.path.join(location_dir, img_name)

        profile = {
            "driver":    "GTiff",
            "dtype":     "uint16",
            "width":     PATCH_PX,
            "height":    PATCH_PX,
            "count":     7,
            "crs":       "EPSG:4326",
            "transform": transform,
            "compress":  "deflate",
            "predictor": 2,
            "tiled":     False,
        }
        with rio.open(img_path, "w", **profile) as dst:
            dst.write(arr)

        meta_rows.append({
            "split": "custom",
            "roi_id": location_name,
            "s2_id": scene_id,
            "img": img_name,
            "date": date_str,
        })
        print(f"  Saved: {img_path}")

    return meta_rows


def save_metadata(meta_rows, location_name="loc", path="out_data"):
    meta_path = os.path.join(path, location_name, "metadata.csv")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "split", "roi_id", "s2_id", "img", "date"],
        )
        writer.writeheader()
        for i, row in enumerate(meta_rows):
            writer.writerow({"index": i, **row})
    print(f"metadata.csv -> {path}") 