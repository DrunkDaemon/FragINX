# FragINX — AI-Powered Perfume Discovery

A local web app that identifies the closest matching perfume in your own labeled image catalog. It uses CLIP image embeddings and cosine similarity, with FastAPI for the backend and a responsive HTML/CSS/JS frontend.

> Important: this starter does **not** ship with thousands of copyrighted product images or claim to be trained on them. It compares an uploaded photo against images you add to your catalog. Results improve as the catalog gains multiple angles, lighting conditions, and bottle variants for each product. Similarity is not a calibrated probability.

## Requirements
- Python 3.10–3.12 recommended
- Internet on first model run so Hugging Face can download `openai/clip-vit-base-patch32` (about 600 MB of model files; exact download can vary)
- About 2–4 GB free disk space recommended for environment and model cache

## Quick start on macOS (VS Code or Terminal)

```bash
cd scoutai-perfume
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The first indexing/prediction can take a while while the model downloads. API docs: http://127.0.0.1:8000/docs.

You can also run `./run_mac.command` from Terminal to install dependencies and start the app in one step.

## Add labeled images
1. Open the **Catalog** tab in the website.
2. Enter the exact brand and perfume name.
3. Upload one or more product photos, ideally multiple angles/backgrounds of the same bottle.
4. Click **Add to catalog**. The backend saves the images and rebuilds the visual index.
5. Open **Identify** and upload an image to find the closest catalog matches.

You can also populate the catalog manually with this folder layout:

```text
data/catalog/
  Dior/
    Sauvage_Eau_de_Toilette/
      front.jpg
      side.jpg
  Chanel/
    Bleu_de_Chanel_Eau_de_Parfum/
      front.jpg
```

Then restart the app or click **Rebuild index** on the Catalog tab. The folder names become the brand and perfume labels; underscores display as spaces.

## What the system does
- Encodes images using a pretrained CLIP vision model.
- Stores one normalized embedding per catalog image.
- Embeds the query image and ranks catalog images by cosine similarity.
- Returns up to three unique product candidates with their best matching catalog image.

This is an image-retrieval baseline, not a newly fine-tuned perfume-specific classifier. It can identify products only if they (or visually similar examples) exist in the catalog. Bottle lookalikes, limited editions, low-quality photos, and hidden labels can cause errors. A real large-scale release should include a legally sourced dataset, train/validation/test splits, per-brand and per-product evaluation, and a calibrated unknown-product threshold.

## Dataset and Attribution

FragINX uses two separate data sources:

- **Perfume images:** The original local image catalog is excluded from this repository because of its size. Users can build their own image catalog or configure a smaller dataset for deployment.
- **Fragrance profiles:** `data/fragrance_profiles.csv` contains structured perfume information, including fragrance families, seasons, times of day, and note pyramids where available.

The fragrance profile dataset is sourced from [Scento's Perfume Database](https://github.com/Scento-com/perfume-dataset), licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

**Attribution:** Scento — [Scento Perfume Database](https://www.scento.com/perfume-database).

Empty fields indicate that information was not recorded in the source dataset. FragINX does not invent missing fragrance notes, longevity, or sillage ratings.

The CLIP model is pretrained and used for visual similarity retrieval; it is not fine-tuned specifically for perfume recognition.


## API
- `GET /api/health` — health and image count
- `GET /api/catalog` — catalog summary
- `POST /api/predict` — multipart field `file`
- `POST /api/catalog/add` — multipart fields `brand`, `perfume`, and one or more `files`
- `POST /api/catalog/reindex` — rebuild the embedding index
