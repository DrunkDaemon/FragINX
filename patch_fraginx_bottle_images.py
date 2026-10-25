from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path.cwd()
MAIN = ROOT / "app" / "main.py"
HTML = ROOT / "static" / "index.html"
MAP = ROOT / "static" / "perfume-images" / "map.json"

if not MAIN.is_file() or not HTML.is_file():
    raise SystemExit("Run this script from the scoutai-perfume project root (the folder containing app/ and static/).")
if not MAP.is_file():
    raise SystemExit("Missing static/perfume-images/map.json. Run prepare_fraginx_images.py first.")

main_text = MAIN.read_text(encoding="utf-8")
html_text = HTML.read_text(encoding="utf-8")
changed_main = False
changed_html = False

# Add the image map load directly after the profile load in the browse endpoint.
old_profiles = '    profiles = load_fragrance_profiles()\n    query = normalize_name(q)'
new_profiles = '''    profiles = load_fragrance_profiles()
    image_map_path = STATIC / "perfume-images" / "map.json"
    try:
        image_map = json.loads(image_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        image_map = {}
    query = normalize_name(q)'''
if 'image_map_path = STATIC / "perfume-images" / "map.json"' not in main_text:
    if old_profiles not in main_text:
        raise SystemExit("Couldn't find the expected browse_fragrances() profile-loading lines in app/main.py. No files changed.")
    main_text = main_text.replace(old_profiles, new_profiles, 1)
    changed_main = True

old_image = '            "image": None, "source_url": (row.get("url") or "").strip()'
new_image = '''            "image": image_map.get(normalize_name(brand) + "|" + normalize_name(name)),
            "source_url": (row.get("url") or "").strip()'''
if '"image": image_map.get(normalize_name(brand)' not in main_text:
    if old_image not in main_text:
        raise SystemExit("Couldn't find the expected 'image': None line in app/main.py. No files changed.")
    main_text = main_text.replace(old_image, new_image, 1)
    changed_main = True

old_card = '''        const bottle = document.createElement('div');
        bottle.className = 'browse-bottle';
        bottle.textContent = (item.brand || 'F').trim().charAt(0).toUpperCase();'''
new_card = '''        const bottle = document.createElement('div');
        bottle.className = 'browse-bottle';
        if (item.image) {
          const img = document.createElement('img');
          img.src = item.image;
          img.alt = `${item.brand || ''} ${item.perfume || ''}`.trim();
          img.loading = 'lazy';
          img.onerror = () => {
            img.remove();
            bottle.textContent = (item.brand || 'F').trim().charAt(0).toUpperCase();
          };
          bottle.appendChild(img);
        } else {
          bottle.textContent = (item.brand || 'F').trim().charAt(0).toUpperCase();
        }'''
if "img.src = item.image;" not in html_text:
    if old_card not in html_text:
        raise SystemExit("Couldn't find the expected browse-card bottle placeholder in static/index.html. No files changed.")
    html_text = html_text.replace(old_card, new_card, 1)
    changed_html = True

# Ensure bottle images fill the existing visual area without cropping.
if ".browse-bottle img" not in html_text:
    style_close = "</style>"
    css = '''
    .browse-bottle img { width: 100%; height: 100%; object-fit: contain; display: block; }
    '''
    if style_close not in html_text:
        raise SystemExit("Couldn't find </style> in static/index.html. No files changed.")
    html_text = html_text.replace(style_close, css + style_close, 1)
    changed_html = True

if changed_main or changed_html:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if changed_main:
        shutil.copy2(MAIN, MAIN.with_name(f"main.py.bak-images-{stamp}"))
        MAIN.write_text(main_text, encoding="utf-8")
    if changed_html:
        shutil.copy2(HTML, HTML.with_name(f"index.html.bak-images-{stamp}"))
        HTML.write_text(html_text, encoding="utf-8")

print("FragINX bottle-image patch complete.")
print("Backend changed:", changed_main)
print("Frontend changed:", changed_html)
print("Image map exists:", MAP.is_file())
print("Next: restart/refresh local FragINX and test /api/fragrances?per_page=5.")
