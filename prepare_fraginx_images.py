
import csv
import hashlib
import json
import re
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "data/fragrance_profiles.csv"
DATASET = ROOT / "data/perfume_dataset/perfumes.csv"
IMAGES = ROOT / "data/perfume_dataset/images/images"
OUTPUT = ROOT / "static/perfume-images"
MAX_IMAGES = 120

def normalize(value):
    value = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return " ".join(value.split())

def main():
    for path in (PROFILES, DATASET, IMAGES):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    with PROFILES.open(encoding="utf-8-sig", newline="") as f:
        profiles = list(csv.DictReader(f))

    with DATASET.open(encoding="utf-8-sig", newline="") as f:
        dataset = list(csv.DictReader(f, delimiter="|"))

    lookup = {}
    for row in dataset:
        key = (normalize(row["brand"]), normalize(row["name_perfume"]))
        filename = row.get("image_name", "").strip()
        path = IMAGES / filename
        if all(key) and filename and path.is_file():
            lookup.setdefault(key, path)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_map = {}
    total_bytes = 0

    def popularity(row):
        try:
            return int(row.get("popularity_rank", ""))
        except (ValueError, TypeError):
            return 10**9

    profiles.sort(key=popularity)

    for profile in profiles:
        brand = profile.get("house", "")
        name = profile.get("name", "")
        key = (normalize(brand), normalize(name))

        if not all(key) or key in image_map or key not in lookup:
            continue

        filename = hashlib.sha1("|".join(key).encode()).hexdigest()[:16] + ".jpg"
        destination = OUTPUT / filename

        try:
            with Image.open(lookup[key]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((480, 480))
                image.save(destination, "JPEG", quality=80, optimize=True)
        except Exception as error:
            print("Skipping", name, error)
            continue

        image_map["|".join(key)] = "/static/perfume-images/" + filename
        total_bytes += destination.stat().st_size

        if len(image_map) >= MAX_IMAGES:
            break

    (OUTPUT / "map.json").write_text(
        json.dumps(image_map, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Prepared {len(image_map)} matched images.")
    print(f"Total image size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Image mapping: {OUTPUT / 'map.json'}")

if __name__ == "__main__":
    main()
