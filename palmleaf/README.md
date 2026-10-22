# Tamil Palm-Leaf Manuscript Digitizer & Restoration System

An AI-powered web application that takes a damaged Tamil palm-leaf manuscript image and performs:
- Image preprocessing (grayscale, blur, adaptive threshold, noise removal)
- Deep learning-based image restoration (lightweight U-Net)
- Tamil OCR using Tesseract
- Tamil text correction and cleanup

---

## Project Structure

```
palmleaf/
├── app/
│   ├── templates/index.html      # Web UI
│   ├── static/
│   │   ├── css/style.css         # Stylesheet
│   │   └── uploads/              # Saved output images
│   └── app.py                    # Flask app + pipeline orchestration
├── models/
│   ├── restoration_model.py      # U-Net architecture
│   ├── train.py                  # Training script
│   └── restorer.pth              # Saved weights (after training)
├── preprocessing/
│   └── preprocess.py             # OpenCV preprocessing pipeline
├── ocr/
│   └── ocr.py                    # Tesseract OCR wrapper
├── utils/
│   └── helpers.py                # Image I/O, base64, text correction
├── data/                         # Place your training images here
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Python 3.11 (64-bit)
This project pins PyTorch 2.3 and Pillow 10.3. Use Python 3.11 for the pinned dependencies; Python 3.14 is too new for these versions. Download Python from https://www.python.org/downloads/

### 2. Tesseract OCR with Tamil language pack

**Windows:**
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. During install, check "Additional language data" → select **Tamil (tam)**
3. Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. Add to PATH, or set in `ocr/ocr.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-tam

# macOS
brew install tesseract
brew install tesseract-lang  # includes Tamil
```

### 3. Hosted Tamil modernization model

PalmLeaf sends OCR text to GroqCloud's hosted `qwen/qwen3.8-27b` model for Tamil correction and ancient-to-modern suggestions. Create an API key in the [GroqCloud console](https://console.groq.com/keys), then add it to the project-root `.env` file:

```env
GROQ_API_KEY=your-groq-api-key
GROQ_API_URL=https://api.groq.com/openai/v1
MODERN_TAMIL_MODEL=qwen/qwen3.8-27b
MODERN_TAMIL_TIMEOUT=150
```

Keep the key server-side; do not put it in browser code or commit `.env`. Groq's Free plan has model-specific rate limits. If the key is missing or the service is unavailable, PalmLeaf still returns conservatively corrected OCR and known phrase rules. Review highlighted words before relying on a reading. Model confidence shown in the UI is an estimated text-quality indicator, not a calibrated probability.

### 4. MongoDB

Start the local MongoDB server. By default, Palmleaf connects to `mongodb://localhost:27017/tamil_manuscripts_db`. Set `MONGO_URI` in `.env` if your MongoDB connection uses a different address. The app creates collections and indexes as needed.

To add the project's sourced Tamil Digital Library catalog references to the repository, run:

```bash
python -m database.seed_tdl_catalog
```

This repeatable import stores a curated set of literary listings and manuscript-catalog references, with source links, in `tdl_catalog`. It does not copy book text or manuscript images.

To add the 41 work-level listings from the CICT Digital Archive, run:

```bash
python -m database.seed_cict_catalog
```

This repeatable import stores titles, collection names, source-institution and pothi counts, and links in `cict_catalog`. These are archive listings, not 41 downloaded manuscripts; images and texts remain at CICT.

To create three clearly marked, source-linked catalog examples for the repository and analytics pages, run:

```bash
python -m database.seed_demo_manuscripts
```

The examples are not digitized scans and contain no OCR text or model predictions. The importer is safe to rerun; each entry is matched to its CICT source record.

---

## Setup

### Step 1 — Create and activate virtual environment
```bash
cd palmleaf
python -m venv venv

# Windows (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

> For CPU-only PyTorch (faster install, no GPU needed):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install flask werkzeug opencv-python numpy pytesseract Pillow pypdfium2
> ```

---

## Training the Model

### Option A — Train on synthetic data (no real images needed)
```bash
python models/train.py
```
This generates synthetic manuscript-like images, trains the U-Net for 30 epochs,
and saves weights to `models/restorer.pth`. Takes ~2–5 minutes on CPU.

### Option B — Train on your own images
1. Place grayscale manuscript images (PNG/JPG) into the `data/` folder
2. Run `python models/train.py`

The script auto-detects images in `data/` and uses them as clean targets,
generating damaged versions on-the-fly for training.

---

## Running the Web App

```bash
python app/app.py
```

Run this from the project root. This starts the full application and connects to MongoDB.

Open your browser at: **http://localhost:5000**

---

## Using the App

1. Choose a JPG, PNG, TIFF, BMP, or PDF scan (for PDFs, page 1 is analyzed and the original PDF is preserved)
2. Click **"Analyse & Restore"**

PDF uploads are preserved as PDFs, with page 1 converted to an image for the existing restoration and OCR pipeline. The original PDF is linked from the saved manuscript detail page; install dependencies from `requirements.txt` to enable PDF conversion.
3. The app displays:
   - Original image
   - Preprocessed image (thresholded/denoised)
   - Restored image (U-Net output)
   - Extracted Tamil text (OCR + correction)
4. Use **"Copy Text"** to copy the extracted Tamil text

Output images are also saved to `app/static/uploads/`.

---

## Notes

- The model runs on CPU by default; GPU is used automatically if available (CUDA)
- OCR accuracy depends on image quality and Tesseract's Tamil model
- For best results, use high-resolution scans (300+ DPI)
- The U-Net is lightweight (features: 32→64→128) to run on any laptop

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `TesseractNotFoundError` | Set `tesseract_cmd` path in `ocr/ocr.py` |
| `No Tamil text detected` | Ensure Tamil lang pack is installed: `tesseract --list-langs` |
| `No trained model found` | Run `python models/train.py` first |
| Slow inference | Normal on CPU; model is small (~1M params) |
| Import errors | Ensure venv is activated and `pip install -r requirements.txt` was run |
