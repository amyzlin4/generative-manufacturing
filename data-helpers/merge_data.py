"""
Merge DeepMS-converted data with existing gmdl.py training data.

Usage:
  python merge_data.py --existing data/train.json --new data/deepms_converted.json --output data/train_merged.json
"""
import json, argparse
from collections import Counter

def merge(existing_path, new_path, output_path):
    with open(existing_path) as f:
        existing = json.load(f)
    with open(new_path) as f:
        new_samples = json.load(f)

    merged = existing + new_samples
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(existing)} + {len(new_samples)} = {len(merged)} samples")
    labels = [s["process_label"] for s in merged]
    print(f"Combined label distribution: {dict(Counter(labels))}")

    for i, name in enumerate(["3-axis CNC", "5-axis CNC", "Injection molding",
                               "Casting", "Forging", "Lathing/Turning",
                               "Sheet metal", "3D printing", "Sintering"]):
        count = labels.count(i)
        print(f"  {i}: {name} → {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", default="data/train.json")
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", default="data/train_merged.json")
    args = parser.parse_args()
    merge(args.existing, args.new, args.output)
