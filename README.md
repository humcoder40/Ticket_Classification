# Complaint Classification System

AI-powered web app for classifying consumer complaints into **product** or **issue** categories. Five models per target (Naive Bayes, Logistic Regression, Random Forest, SVM, DistilBERT) are combined with a **soft-voting ensemble**. Optional **Groq Whisper** voice input transcribes spoken complaints in the browser.

## Features

- **Final ensemble prediction** — soft voting across all successful models, with majority-vote metadata
- **Analyze both** — run product and issue classification on the same text in one click
- **Sample complaints** — quick-fill dropdown for demo and grading
- **Low-confidence warning** — banner when ensemble confidence is low or models disagree
- **Download JSON** — export the latest prediction results
- **Voice input** — microphone → Groq `whisper-large-v3-turbo` transcription
- **Model status panel** — shows loaded TF-IDF and BERT models per target

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd FinalProject
pip install -r requirements.txt
```

### 2. Add models and data (not in GitHub)

This repository **does not** include trained artifacts or the raw dataset (~1 GB). After cloning, you need:

| Artifact | Location | How to create |
|----------|----------|---------------|
| TF-IDF models | `models/` | Run `train_classification_model.ipynb` |
| BERT models | `bert_model_product/`, `bert_model_issue/` | Run `python train_bert_models.py` |
| Training data | `complaints-2021-05-14_08_16.json` | Download separately (CFPB-style complaint JSON) |

Expected TF-IDF files per target (`product`, `issue`):

- `{target}_tfidf_vectorizer.pkl`
- `{target}_label_encoder.pkl`
- `{target}_naive_bayes_model.pkl`, `{target}_logistic_regression_model.pkl`, etc.

### 3. Environment (optional voice)

```bash
copy .env.example .env
```

Add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Voice input works without the key; only text analysis is required.

### 4. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage

1. Pick **Product** or **Issue**, or use **Analyze Both**.
2. Enter text, pick a **sample complaint**, or use **Voice Input**.
3. Click **Analyze Complaint** (or **Analyze Both**).
4. Review the **Final Ensemble Prediction**, individual model cards, and any low-confidence warning.
5. Click **Download JSON** to save results.

Keyboard shortcut: **Ctrl+Enter** runs a single-target analysis.

## API

### `POST /predict`

```json
{ "text": "Your complaint...", "target": "product" }
```

Returns `final_ensemble_prediction`, per-model `results`, and `timestamp`.

### `POST /transcribe`

Multipart form field `audio` (WebM/OGG/WAV, max 25 MB). Requires `GROQ_API_KEY`.

### `GET /models/status`

Loaded models per target and transcription availability.

See [README_UI.md](README_UI.md) for full API examples and troubleshooting.

## Project structure

```
FinalProject/
├── app.py                          # Flask API, ensemble, model loading
├── train_bert_models.py            # Full BERT training
├── test_bert_models.py             # BERT evaluation
├── reexport_sklearn_pickles.py     # Re-save pickles after sklearn upgrade
├── train_classification_model.ipynb
├── requirements.txt
├── .env.example
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── models/                         # gitignored — TF-IDF pickles
├── bert_model_product/             # gitignored
├── bert_model_issue/               # gitignored
└── complaints-*.json               # gitignored — training data
```

## Training notes

- **TF-IDF models**: notebook trains four sklearn classifiers per target.
- **BERT**: `train_bert_models.py` fine-tunes DistilBERT (~72% product / ~67% issue holdout accuracy on full dataset).
- **sklearn version**: if you see pickle version warnings, run `python reexport_sklearn_pickles.py` and restart the app.

## What to commit vs ignore

`.gitignore` excludes:

- `.env` (secrets)
- `models/`, `bert_model_*` (large binaries)
- `complaints-*.json` (dataset)
- `__pycache__/`, virtualenvs, IDE files

Commit source code, templates, static assets, notebooks, scripts, and documentation.

## License / course

University machine learning final project — adjust license and attribution as required by your course.
