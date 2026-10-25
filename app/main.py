from pathlib import Path
from typing import Optional
import csv
import io
import json
import shutil
import threading
import uuid
import urllib.request
import urllib.error
import unicodedata

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
CATALOG = DATA / "catalog"
UPLOADS = DATA / "uploads"
DATASET_DIR = DATA / "perfume_dataset"
DATASET_IMAGES = DATASET_DIR / "images"
DATASET_CSV = DATASET_DIR / "perfumes.csv"
INDEX_FILE = DATA / "catalog_index.json"
VECTOR_FILE = DATA / "catalog_vectors.npy"
FRAGRANCE_PROFILES_FILE = DATA / "fragrance_profiles.csv"
SCENTO_PROFILES_URL = "https://raw.githubusercontent.com/Scento-com/perfume-dataset/main/perfumes.csv"
_profile_cache = None
_profile_lock = threading.Lock()
for folder in (STATIC, CATALOG, UPLOADS, DATASET_IMAGES):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FragINX — Fragrance Intelligence", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/catalog", StaticFiles(directory=CATALOG), name="catalog")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/dataset-images", StaticFiles(directory=DATASET_IMAGES), name="dataset-images")

_model = None
_processor = None
_model_lock = threading.Lock()
_index_lock = threading.Lock()
_catalog_items = []
_catalog_vectors = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _model_lock:
        if _model is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
                model_id = "openai/clip-vit-base-patch32"
                _processor = CLIPProcessor.from_pretrained(model_id)
                _model = CLIPModel.from_pretrained(model_id)
                _model.eval()
            except Exception as exc:
                raise RuntimeError(
                    "Could not load the CLIP vision model. On first run, make sure you have internet access and installed requirements. "
                    f"Details: {exc}"
                ) from exc
    return _model, _processor


def image_embedding(image: Image.Image) -> np.ndarray:
    import torch
    model, processor = load_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().numpy().astype(np.float32)


def safe_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None



def rebuild_index():
    global _catalog_items, _catalog_vectors

    items = []
    vectors = []

    if not DATASET_CSV.exists():
        raise RuntimeError(f"Dataset CSV not found: {DATASET_CSV}")

    with DATASET_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")

        for row in reader:
            filename = (row.get("image_name") or "").strip()
            brand = (row.get("brand") or "").strip()
            perfume = (row.get("name_perfume") or "").strip()
            image_path = DATASET_IMAGES / "images" / filename

            if not filename or not brand or not perfume or not image_path.is_file():
                continue

            items.append({
                "brand": brand,
                "perfume": perfume,
                "image": "/dataset-images/images/" + filename,
                "path": str(image_path),
            })

    for image_path in sorted(CATALOG.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp"
        }:
            continue

        relative = image_path.relative_to(CATALOG)

        if len(relative.parts) < 3:
            continue

        image = safe_image(image_path)
        if image is None:
            continue

        items.append({
            "brand": relative.parts[0].replace("_", " "),
            "perfume": relative.parts[1].replace("_", " "),
            "image": "/catalog/" + relative.as_posix(),
            "path": str(image_path),
        })

    metadata_file = DATA / "catalog_index.json"
    cache_file = VECTOR_FILE

    signature = {
        "csv_mtime": DATASET_CSV.stat().st_mtime_ns,
        "csv_size": DATASET_CSV.stat().st_size,
        "catalog": [
            (item["path"], Path(item["path"]).stat().st_mtime_ns)
            for item in items
            if str(CATALOG) in item["path"]
        ],
    }

    if metadata_file.exists() and cache_file.exists():
        try:
            cached = json.loads(metadata_file.read_text(encoding="utf-8"))
            cached_vectors = np.load(cache_file, allow_pickle=False)

            if (
                cached.get("signature") == signature
                and len(cached["items"]) == len(items)
                and cached_vectors.shape[0] == len(items)
            ):
                with _index_lock:
                    _catalog_items = cached["items"]
                    _catalog_vectors = cached_vectors
                return len(_catalog_items)
        except Exception:
            pass

    print(f"Building visual index for {len(items)} images...")

    indexed_items = []
    indexed_vectors = []

    for i, item in enumerate(items, start=1):
        image = safe_image(Path(item["path"]))
        if image is None:
            continue

        try:
            vector = image_embedding(image)
        except Exception as exc:
            print(f"Skipping {item['path']}: {exc}")
            continue

        indexed_items.append(item)
        indexed_vectors.append(vector)

        if i % 250 == 0:
            print(f"Processed {i}/{len(items)} images")

    matrix = (
        np.stack(indexed_vectors).astype(np.float32)
        if indexed_vectors else None
    )

    if matrix is None:
        raise RuntimeError("No valid perfume images could be indexed.")

    saved_items = [
        {
            "brand": item["brand"],
            "perfume": item["perfume"],
            "image": item["image"],
            "path": item["path"],
        }
        for item in indexed_items
    ]

    metadata_file.write_text(
        json.dumps({
            "signature": signature,
            "items": saved_items,
        }),
        encoding="utf-8",
    )
    np.save(cache_file, matrix)

    with _index_lock:
        _catalog_items = saved_items
        _catalog_vectors = matrix

    print(f"Index complete: {len(saved_items)} images")
    return len(saved_items)



def get_catalog_count():
    return len(_catalog_items)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join("".join(char.lower() if char.isalnum() else " " for char in value).split())


def split_profile_values(value):
    if not value:
        return []
    return [part.strip() for part in value.replace("|", ";").split(";") if part.strip()]


def load_fragrance_profiles():
    """Load the optional CC BY 4.0 Scento CSV; download once when possible."""
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache
    with _profile_lock:
        if _profile_cache is not None:
            return _profile_cache
        if not FRAGRANCE_PROFILES_FILE.exists():
            try:
                request = urllib.request.Request(
                    SCENTO_PROFILES_URL,
                    headers={"User-Agent": "FragINX/1.1 (local fragrance profile lookup)"},
                )
                with urllib.request.urlopen(request, timeout=8) as response:
                    payload = response.read()
                DATA.mkdir(parents=True, exist_ok=True)
                FRAGRANCE_PROFILES_FILE.write_bytes(payload)
                print("Downloaded the Scento fragrance profile dataset (CC BY 4.0).")
            except Exception as exc:
                print(f"External fragrance profiles unavailable; using local catalog metadata: {exc}")
                _profile_cache = []
                return _profile_cache
        try:
            with FRAGRANCE_PROFILES_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
                _profile_cache = list(csv.DictReader(handle))
        except Exception as exc:
            print(f"Could not read fragrance profile dataset: {exc}")
            _profile_cache = []
    return _profile_cache


def local_dataset_profile(brand: str, perfume: str):
    if not DATASET_CSV.exists():
        return None
    target_brand, target_perfume = normalize_name(brand), normalize_name(perfume)
    try:
        with DATASET_CSV.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="|"):
                if normalize_name(row.get("brand", "")) == target_brand and normalize_name(row.get("name_perfume", "")) == target_perfume:
                    ingredients = split_profile_values(row.get("ingredients", ""))
                    accords = split_profile_values(row.get("fragrances", ""))
                    return {
                        "brand": row.get("brand", ""),
                        "perfume": row.get("name_perfume", ""),
                        "family": row.get("family") or "",
                        "secondary_family": row.get("subfamily") or "",
                        "top_notes": [], "heart_notes": [], "base_notes": [],
                        "notes_available": False,
                        "all_notes": ingredients,
                        "accords": accords,
                        "seasons": [], "times_of_day": [],
                        "description": row.get("description") or "",
                        "longevity": None, "sillage": None,
                        "source_name": "Local perfume image dataset",
                        "source_url": None,
                    }
    except Exception as exc:
        print(f"Could not read local perfume metadata: {exc}")
    return None


def find_fragrance_profile(brand: str, perfume: str):
    profiles = load_fragrance_profiles()
    target_brand, target_perfume = normalize_name(brand), normalize_name(perfume)
    best = None
    best_score = 0
    for row in profiles:
        row_brand = row.get("house", "")
        row_perfume = row.get("name", "")
        brand_score = 2 if normalize_name(row_brand) == target_brand else 0
        perfume_score = 3 if normalize_name(row_perfume) == target_perfume else 0
        if not perfume_score:
            continue
        score = brand_score + perfume_score
        if score > best_score:
            best, best_score = row, score
    if best and best_score >= 3:
        top = split_profile_values(best.get("top_notes", ""))
        heart = split_profile_values(best.get("heart_notes", ""))
        base = split_profile_values(best.get("base_notes", ""))
        return {
            "brand": best.get("house", brand),
            "perfume": best.get("name", perfume),
            "year": best.get("year") or None,
            "concentration": best.get("concentration") or None,
            "family": best.get("family") or None,
            "secondary_family": best.get("secondary_family") or None,
            "character": split_profile_values(best.get("character", "")),
            "top_notes": top, "heart_notes": heart, "base_notes": base,
            "notes_available": bool(top or heart or base),
            "all_notes": split_profile_values(best.get("notes", "")),
            "accords": [],
            "seasons": split_profile_values(best.get("seasons", "")),
            "times_of_day": split_profile_values(best.get("times_of_day", "")),
            "description": best.get("description") or "",
            "longevity": None, "sillage": None,
            "source_name": "Scento Perfume Database (CC BY 4.0)",
            "source_url": best.get("url") or "https://www.scento.com/perfume-database",
        }
    return local_dataset_profile(brand, perfume)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "catalog_images": get_catalog_count(), "model": "OpenAI CLIP ViT-B/32", "brand": "FragINX", "fragrance_profiles": len(load_fragrance_profiles())}


@app.get("/api/perfume/profile")
def perfume_profile(brand: str, perfume: str):
    profile = find_fragrance_profile(brand, perfume)
    if profile is None:
        return {
            "brand": brand, "perfume": perfume,
            "top_notes": [], "heart_notes": [], "base_notes": [],
            "notes_available": False, "all_notes": [], "accords": [],
            "seasons": [], "times_of_day": [], "family": None,
            "secondary_family": None, "character": [], "description": "",
            "year": None, "concentration": None, "longevity": None,
            "sillage": None, "source_name": None, "source_url": None,
        }
    return profile


@app.get("/api/perfume/search")
def perfume_search(q: str = ""):
    query = normalize_name(q)
    if len(query) < 2:
        return {"results": []}
    profiles = load_fragrance_profiles()
    results = []
    for row in profiles:
        name = row.get("name", "")
        brand = row.get("house", "")
        if query in normalize_name(name) or query in normalize_name(brand):
            results.append({"brand": brand, "perfume": name, "family": row.get("family") or "", "image": None, "source_url": row.get("url") or ""})
            if len(results) >= 20:
                break
    return {"results": results}


@app.get("/api/fragrances")
def browse_fragrances(q: str = "", family: str = "", season: str = "",
                      time_of_day: str = "", page: int = 1, per_page: int = 24):
    profiles = load_fragrance_profiles()
    image_map_path = Path(__file__).resolve().parent.parent / "static" / "perfume-images" / "map.json"
    try:
        image_map = json.loads(image_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        image_map = {}
    query = normalize_name(q)
    family_filter = normalize_name(family)
    season_filter = normalize_name(season)
    time_filter = normalize_name(time_of_day)
    families = sorted({r.get("family", "").strip() for r in profiles if r.get("family", "").strip()}, key=str.casefold)
    seasons = sorted({v for r in profiles for v in split_profile_values(r.get("seasons", ""))}, key=str.casefold)
    times = sorted({v for r in profiles for v in split_profile_values(r.get("times_of_day", ""))}, key=str.casefold)
    matches = []
    for row in profiles:
        name = (row.get("name") or "").strip()
        brand = (row.get("house") or "").strip()
        row_family = (row.get("family") or "").strip()
        row_seasons = split_profile_values(row.get("seasons", ""))
        row_times = split_profile_values(row.get("times_of_day", ""))
        searchable = normalize_name(" ".join([name, brand, row_family,
            row.get("secondary_family", ""), row.get("character", ""),
            row.get("top_notes", ""), row.get("heart_notes", ""),
            row.get("base_notes", ""), row.get("notes", "")]))
        if query and query not in searchable:
            continue
        if family_filter and family_filter not in normalize_name(row_family):
            continue
        if season_filter and not any(season_filter in normalize_name(v) for v in row_seasons):
            continue
        if time_filter and not any(time_filter in normalize_name(v) for v in row_times):
            continue
        matches.append({
            "brand": brand, "perfume": name,
            "year": (row.get("year") or "").strip() or None,
            "concentration": (row.get("concentration") or "").strip() or None,
            "family": row_family,
            "secondary_family": (row.get("secondary_family") or "").strip(),
            "top_notes": split_profile_values(row.get("top_notes", "")),
            "heart_notes": split_profile_values(row.get("heart_notes", "")),
            "base_notes": split_profile_values(row.get("base_notes", "")),
            "seasons": row_seasons, "times_of_day": row_times,
            "description": (row.get("description") or "").strip(),
                        "image": image_map.get(
                normalize_name(brand) + "|" + normalize_name(name)
            ),
            "source_url": (row.get("url") or "").strip()
        })
    page = max(1, page)
    per_page = min(48, max(1, per_page))
    total = len(matches)
    start = (page - 1) * per_page
    return {
        "results": matches[start:start + per_page], "total": total,
        "page": page, "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "families": families, "seasons": seasons, "times_of_day": times,
        "source": "Scento Perfume Database (CC BY 4.0)"
    }


def build_catalog_products():
    products = {}
    for item in _catalog_items:
        key = (item["brand"], item["perfume"])
        if key not in products:
            products[key] = {
                "brand": item["brand"],
                "perfume": item["perfume"],
                "image": item["image"],
                "image_count": 0,
            }
        products[key]["image_count"] += 1
    return list(products.values())


@app.get("/api/catalog")
def catalog():
    products = build_catalog_products()
    # Keep the existing catalog-management response lightweight.
    return {
        "image_count": len(_catalog_items),
        "product_count": len(products),
        "products": products[:100],
    }


@app.get("/api/catalog/browse")
def browse_catalog(q: str = "", page: int = 1, per_page: int = 24):
    """Search and paginate every product in the visual catalog."""
    products = build_catalog_products()
    query = normalize_name(q)
    if query:
        products = [
            item for item in products
            if query in normalize_name(item["brand"])
            or query in normalize_name(item["perfume"])
            or query in normalize_name(item["brand"] + " " + item["perfume"])
        ]

    page = max(1, page)
    per_page = min(100, max(1, per_page))
    total = len(products)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)
    start = (page - 1) * per_page
    return {
        "image_count": len(_catalog_items),
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
        "products": products[start:start + per_page],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is too large. Maximum size is 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file could not be read as an image.")
    if _catalog_vectors is None or not _catalog_items:
        raise HTTPException(400, "Your catalog is empty. Add a few labeled perfume images in the Catalog tab first.")
    try:
        query = image_embedding(image)
    except Exception as exc:
        raise HTTPException(503, str(exc))
    similarities = _catalog_vectors @ query
    top_indices = np.argsort(similarities)[::-1][:5]
    matches = []
    seen = set()
    for idx in top_indices:
        item = _catalog_items[int(idx)]
        key = (item["brand"], item["perfume"])
        if key in seen:
            continue
        seen.add(key)
        score = float(similarities[int(idx)])
        matches.append({"brand": item["brand"], "perfume": item["perfume"], "image": item["image"], "similarity": round(max(0.0, min(1.0, score)) * 100, 1)})
        if len(matches) == 3:
            break
    query_name = f"query-{uuid.uuid4().hex[:10]}.jpg"
    image.save(UPLOADS / query_name, quality=90)
    return {"query_image": "/uploads/" + query_name, "matches": matches, "method": "CLIP visual embedding similarity", "note": "Similarity is not a calibrated probability. Results depend on the quality and coverage of your catalog."}


@app.post("/api/catalog/add")
async def add_to_catalog(
    brand: str = Form(...),
    perfume: str = Form(...),
    files: list[UploadFile] = File(...),
):
    brand = " ".join(brand.strip().split())
    perfume = " ".join(perfume.strip().split())
    if not brand or not perfume:
        raise HTTPException(400, "Brand and perfume name are required.")
    if len(files) > 30:
        raise HTTPException(400, "Add up to 30 images at a time.")
    safe_brand = "_".join("".join(c for c in brand if c.isalnum() or c in " _-").split())[:80]
    safe_perfume = "_".join("".join(c for c in perfume if c.isalnum() or c in " _-").split())[:100]
    target = CATALOG / safe_brand / safe_perfume
    target.mkdir(parents=True, exist_ok=True)
    saved = 0
    for upload in files:
        raw = await upload.read()
        if not raw:
            continue
        if len(raw) > 12 * 1024 * 1024:
            continue
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        filename = f"{uuid.uuid4().hex[:12]}.jpg"
        im.thumbnail((1400, 1400))
        im.save(target / filename, quality=92, optimize=True)
        saved += 1
    if saved == 0:
        raise HTTPException(400, "No valid images were saved. Please upload JPG, PNG, or WEBP files under 12 MB each.")
    try:
        count = rebuild_index()
    except Exception as exc:
        raise HTTPException(503, f"Images were saved, but indexing failed: {exc}")
    return {"message": f"Added {saved} image(s) for {perfume} by {brand}.", "saved": saved, "catalog_images": count}


@app.post("/api/catalog/reindex")
def reindex():
    try:
        count = rebuild_index()
        return {"message": "Catalog index rebuilt.", "catalog_images": count}
    except Exception as exc:
        raise HTTPException(503, str(exc))


@app.on_event("startup")
def startup():
    if DATASET_CSV.exists():
        try:
            count = rebuild_index()
            print(f"FragINX ready: {count} perfume images indexed.")
        except Exception as exc:
            print(f"Catalog indexing deferred: {exc}")