import json
import random
import sys
import os
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python datasplit.py <input.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON file must contain a top-level array")
        sys.exit(1)

    random.shuffle(data)

    split_idx = int(len(data) * 0.8)
    train = data[:split_idx]
    val = data[split_idx:]

    stem = input_path.stem
    parent = input_path.parent

    train_path = parent / f"train.json"
    val_path = parent / f"validation.json"

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)

    print(f"Wrote {len(train)} items to {train_path}")
    print(f"Wrote {len(val)} items to {val_path}")


if __name__ == "__main__":
    main()
