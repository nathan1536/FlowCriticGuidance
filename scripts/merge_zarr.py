#!/usr/bin/env python3
"""Merge multiple rollout zarr datasets into one."""
import numpy as np
import zarr, sys

inputs = sys.argv[1:-1]  # all args except last
output = sys.argv[-1]     # last arg = output path

all_data = {k: [] for k in ["img", "state", "full_state", "next_full_state", "action", "reward", "done"]}
episode_ends = []
offset = 0

for zpath in inputs:
    root = zarr.open(zpath, mode="r")
    for key in all_data:
        if key in root["data"]:
            all_data[key].append(np.array(root["data"][key]))
    ends = np.array(root["meta"]["episode_ends"])
    episode_ends.append(ends + offset)
    offset += int(ends[-1])
    print(f"  {zpath}: {len(ends)} episodes, {int(ends[-1])} steps")

out = zarr.group(output)
out_data = out.create_group("data")
out_meta = out.create_group("meta")
comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

for key, arrays in all_data.items():
    if arrays:
        merged = np.concatenate(arrays, axis=0)
        chunks = (100,) + merged.shape[1:] if merged.ndim > 1 else (100,)
        out_data.create_dataset(key, data=merged, chunks=chunks, dtype=merged.dtype, compressor=comp)
        print(f"  {key}: {merged.shape}")

merged_ends = np.concatenate(episode_ends)
out_meta.create_dataset("episode_ends", data=merged_ends, dtype="int64", compressor=comp)
print(f"\nMerged {len(inputs)} zarrs → {output} ({len(merged_ends)} episodes, {offset} total steps)")