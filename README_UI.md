# Complaint Classification Web Interface

A modern web interface for the Complaint Classification System: five ML models per target (TF-IDF baselines + BERT) with a soft-voting **Final Ensemble Prediction**.

## Features

- **Final Ensemble Prediction** — Soft voting across all successful models, with majority-vote metadata
- **BERT inference** — DistilBERT models for product and issue classification (GPU when available)
- **Multi-model comparison** — Naive Bayes, Logistic Regression, Random Forest, SVM, and BERT side by side
- **Product / Issue targets** — Switch classification type from the UI
- **Model status panel** — Shows which models are loaded per target
- **Voice input (Groq Whisper)** — Record from the microphone; audio is sent to `POST /transcribe`

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Verify CUDA (optional, for BERT):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Train models (if not already done)

| Artifact | How |
|----------|-----|
| TF-IDF models | Run `train_classification_model.ipynb` |
| BERT models | Run `python train_bert_models.py` (or the BERT cell in the notebook) |

Expected directories:

- `models/` — `{target}_tfidf_vectorizer.pkl`, `{target}_label_encoder.pkl`, `{target}_*_model.pkl`
- `bert_model_product/`, `bert_model_issue/` — HuggingFace weights + `label_encoder.pkl`

### 3. Configure Groq voice (optional)

```bash
copy .env.example .env
```

Add your key to `.env`:

```
GROQ_API_KEY=your_key_here
```

### 4. Re-export pickles (after sklearn upgrade)

If you see `InconsistentVersionWarning` when loading pickles:

```bash
python reexport_sklearn_pickles.py
```

### 5. Run the app

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Usage

1. Choose **Product** or **Issue** classification.
2. Enter complaint text, or click **Voice Input** to record and transcribe.
3. Click **Analyze Complaint**.
4. Read the **Final Ensemble Prediction** at the top, then compare individual model cards below.

## API

### `POST /predict`

Request:

```json
{ "text": "Your complaint...", "target": "product" }
```

Response (abbreviated):

```json
{
  "success": true,
  "target": "product",
  "final_ensemble_prediction": {
    "prediction": "Credit card or prepaid card",
    "confidence": 0.601,
    "method": "soft_voting",
    "models_used": ["Naive Bayes", "Logistic Regression", "Random Forest", "SVM", "BERT"],
    "model_count": 5,
    "majority_vote": { "prediction": "...", "votes": 4, "total": 5 },
    "top_3": [...]
  },
  "results": { "Naive Bayes": {...}, "BERT": {...}, ... },
  "timestamp": "..."
}
```

### `POST /transcribe`

Multipart form upload with field name `audio` (WebM/OGG/WAV, max 25 MB).

Response:

```json
{
  "success": true,
  "text": "Transcribed complaint text...",
  "model": "whisper-large-v3-turbo",
  "timestamp": "..."
}
```

Requires `GROQ_API_KEY` in a `.env` file (see `.env.example`).

### `GET /models/status`

Returns loaded models per target (TF-IDF + BERT flags).

## File structure

```
FinalProject/
├── app.py                          # Flask API, ensemble logic, model loading
├── train_bert_models.py            # Full BERT training
├── test_bert_models.py             # BERT holdout evaluation
├── reexport_sklearn_pickles.py     # Re-save pickles for current sklearn
├── requirements.txt
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── models/                         # TF-IDF pickles (per target)
├── bert_model_product/
├── bert_model_issue/
└── train_classification_model.ipynb
```

## Troubleshooting

### No models found

- Train TF-IDF and/or BERT models and confirm files exist under `models/` and `bert_model_*`.

### sklearn version warnings

- Run `python reexport_sklearn_pickles.py` and restart `app.py`.

### BERT not in results

- Confirm `bert_model_product/` and `bert_model_issue/` exist and `/models/status` shows BERT loaded.

### Voice input not working

- Ensure `.env` contains a valid `GROQ_API_KEY` and restart `python app.py`.
- Allow microphone permission in the browser.
- Check **Model Status** — Voice (Groq) should show `[OK] whisper-large-v3-turbo`.

### Port in use

- Change the port in `app.py`: `app.run(debug=True, port=5001)`.

## Notes

- Set `debug=False` for production.
