import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


def resolve_image_path(image_field: object, images_root: Optional[str], hf_cache: Optional[str]) -> Dict:
    """Try to resolve an image path from a dataset record's `image` field.
    Returns a dict with status and tried candidates for debugging.
    """
    tried: List[str] = []
    if isinstance(image_field, str):
        if images_root:
            tried.append(os.path.join(images_root, image_field))
        tried.append(image_field)
        if hf_cache:
            tried.append(os.path.join(hf_cache, image_field))
        for path in tried:
            if os.path.isfile(path):
                return {"found": True, "path": path, "tried": tried}
        return {"found": False, "path": None, "tried": tried}
    else:
        # Could be dict-like, PIL image (if cast) or bytes; we just report the type
        return {"found": None, "path": None, "tried": tried, "type": str(type(image_field))}


def main():
    dataset_name = os.getenv("DATASET_NAME", "Xkev/LLaVA-CoT-100k")
    dataset_split = os.getenv("DATASET_SPLIT", "train[:10]")
    images_root = os.getenv("IMAGES_ROOT")
    hf_cache = os.getenv("HF_DATASETS_CACHE")

    print(f"Inspecting dataset: {dataset_name} ({dataset_split})")
    print(f"IMAGES_ROOT={images_root}")
    print(f"HF_DATASETS_CACHE={hf_cache}")

    ds = load_dataset(dataset_name, split=dataset_split)
    print(f"Loaded {len(ds)} examples")

    # Print feature schema
    print("\nFeatures:")
    print(ds.features)

    sample_count = min(10, len(ds))
    out: List[Dict] = []
    for i in range(sample_count):
        ex = ds[i]
        rec: Dict = {"idx": i, "keys": list(ex.keys())}
        # Conversations overview
        conv = ex.get("conversations")
        rec["conversations_len"] = len(conv) if isinstance(conv, list) else None
        if isinstance(conv, list) and conv:
            rec["first_turn_keys"] = list(conv[0].keys())
            rec["first_turn_from"] = conv[0].get("from")
            rec["first_turn_value_preview"] = str(conv[0].get("value", ""))[:120]

        # Image field check
        img = ex.get("image")
        rec["image_field_type"] = str(type(img))
        res = resolve_image_path(img, images_root, hf_cache)
        rec["image_resolution"] = res
        out.append(rec)

    output_dir = Path("dataset_check_output")
    output_dir.mkdir(exist_ok=True)
    outfile = output_dir / "inspect_summary.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nWrote inspection summary: {outfile}")
    # Pretty print a couple of rows
    for rec in out[:3]:
        print("\n--- Record ---")
        for k, v in rec.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
