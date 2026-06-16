from flask import Flask, render_template, request, jsonify
from collections import Counter
import pickle
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)

GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
_groq_client = None

# Global variables for models
models = {}  # Structure: {target: {model_name: model}}
bert_models = {}  # Structure: {target: {'model': model, 'tokenizer': tokenizer, 'label_encoder': label_encoder}}
vectorizers = {}  # Structure: {target: vectorizer}
label_encoders = {}  # Structure: {target: label_encoder}
model_names = []  # List of available model names
targets = ['product', 'issue']  # Available prediction targets
MAX_LEN = 256  # Max sequence length for BERT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_groq_client():
    """Return a cached Groq client when GROQ_API_KEY is configured."""
    global _groq_client
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    if _groq_client is None:
        _groq_client = Groq(api_key=api_key)
    return _groq_client

def preprocess_text(text):
    """Basic text cleaning - same as in notebook"""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_bert_models():
    """Load BERT models for both product and issue"""
    global bert_models
    
    # BERT model directories
    bert_model_dirs = {
        'product': 'bert_model_product',
        'issue': 'bert_model_issue',
    }
    
    print("\n" + "=" * 80)
    print("Loading BERT Models")
    print("=" * 80)
    
    for target in targets:
        model_dir = bert_model_dirs.get(target)
        if not model_dir or not os.path.exists(model_dir):
            print(f"[WARN] BERT model directory not found for {target}: {model_dir}")
            continue
        
        try:
            print(f"\nLoading BERT model for: {target.upper()}")
            print(f"  Directory: {model_dir}")
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(DEVICE)
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Load label encoder
            label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            bert_models[target] = {
                'model': model,
                'tokenizer': tokenizer,
                'label_encoder': label_encoder
            }
            
            print(f"[OK] BERT model loaded for {target}")
            print(f"  Number of classes: {len(label_encoder.classes_)}")
            
        except Exception as e:
            print(f"[ERROR] Error loading BERT model for {target}: {e}")
            import traceback
            traceback.print_exc()
    
    return len(bert_models) > 0

def load_models():
    """Load all saved models for both product and issue"""
    global models, vectorizers, label_encoders, model_names
    
    models_dir = 'models'
    
    # Model file mappings
    model_files = {
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'SVM': 'svm_model.pkl'
    }
    
    # Load models for each target
    for target in targets:
        print(f"\nLoading models for: {target.upper()}")
        print("-" * 80)
        
        models[target] = {}
        model_names_for_target = []
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, f'{target}_tfidf_vectorizer.pkl')
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizers[target] = pickle.load(f)
            print(f"[OK] Vectorizer loaded for {target}")
        except Exception as e:
            print(f"Error loading vectorizer for {target}: {e}")
            continue
        
        # Load label encoder
        label_encoder_path = os.path.join(models_dir, f'{target}_label_encoder.pkl')
        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoders[target] = pickle.load(f)
            print(f"[OK] Label encoder loaded for {target}")
        except Exception as e:
            print(f"Error loading label encoder for {target}: {e}")
            continue
        
        # Load all available models for this target
        for model_name, filename_base in model_files.items():
            filename = f'{target}_{filename_base}'
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        models[target][model_name] = pickle.load(f)
                    model_names_for_target.append(model_name)
                    print(f"[OK] {model_name} loaded for {target}")
                except Exception as e:
                    print(f"Error loading {model_name} for {target}: {e}")
        
        if not models[target]:
            print(f"[WARN] No models found for {target}")
        else:
            # Add unique model names to global list
            for name in model_names_for_target:
                if name not in model_names:
                    model_names.append(name)
    
    # Load BERT models
    load_bert_models()
    
    # Add BERT to model names if loaded
    if bert_models:
        if 'BERT' not in model_names:
            model_names.append('BERT')
    
    if not models and not bert_models:
        print("\n[WARN] No models found. Please train models first using the notebook.")
        return False
    
    print("\n" + "=" * 80)
    print("Model Loading Summary:")
    print("=" * 80)
    for target in targets:
        tfidf_count = len(models.get(target, {}))
        bert_count = 1 if target in bert_models else 0
        total = tfidf_count + bert_count
        if total > 0:
            print(f"  {target.upper()}: {total} model(s) loaded ({tfidf_count} TF-IDF + {bert_count} BERT)")
    
    return True

def predict_complaint_bert(text, target='product'):
    """Predict using BERT model for a specific target"""
    if target not in bert_models:
        return None, None, None
    
    bert_data = bert_models[target]
    model = bert_data['model']
    tokenizer = bert_data['tokenizer']
    label_encoder = bert_data['label_encoder']
    
    # Preprocess
    text_processed = preprocess_text(text)
    if not text_processed:
        return None, None, None
    
    try:
        # Tokenize
        encoding = tokenizer(
            text_processed,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predictions
        prediction_encoded = torch.argmax(logits, dim=1).cpu().numpy()[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(probabilities[0][prediction_encoded].cpu().numpy())
        
        # Get top 3 predictions
        probs_np = probabilities[0].cpu().numpy()
        top_3_indices = probs_np.argsort()[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            class_name = label_encoder.classes_[idx]
            prob = float(probs_np[idx])
            top_3_predictions.append({
                'class': class_name,
                'probability': prob
            })
        
        return prediction, confidence, top_3_predictions
    except Exception as e:
        print(f"BERT prediction error for {target}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict_complaint(text, model, target='product'):
    """Predict using a specific model for a specific target"""
    if target not in vectorizers or target not in label_encoders:
        return None, None, None
    
    vectorizer = vectorizers[target]
    label_encoder = label_encoders[target]
    
    if not vectorizer or not label_encoder or not model:
        return None, None, None
    
    # Preprocess
    text_processed = preprocess_text(text)
    if not text_processed:
        return None, None, None
    
    # Vectorize
    try:
        text_vectorized = vectorizer.transform([text_processed])
    except Exception as e:
        print(f"Vectorization error for {target}: {e}")
        return None, None, None
    
    # Predict
    try:
        prediction_encoded = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(prediction_proba[prediction_encoded])
        
        # Get top 3 predictions
        top_3_indices = prediction_proba.argsort()[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            class_name = label_encoder.classes_[idx]
            prob = float(prediction_proba[idx])
            top_3_predictions.append({
                'class': class_name,
                'probability': prob
            })
        
        return prediction, confidence, top_3_predictions
    except Exception as e:
        print(f"Prediction error for {target}: {e}")
        return None, None, None

def compute_ensemble_prediction(results):
    """
    Aggregate successful model outputs using soft voting (averaged class scores).
    Majority vote is included for transparency and tie-breaking context.
    """
    successful = {
        name: result
        for name, result in results.items()
        if isinstance(result, dict) and result.get('prediction') and not result.get('error')
    }
    if not successful:
        return None

    class_scores = {}
    votes = []

    for result in successful.values():
        votes.append(result['prediction'])
        prob_map = {
            item['class']: float(item['probability'])
            for item in result.get('top_3', [])
        }
        prediction = result['prediction']
        confidence = float(result.get('confidence', 0))
        if prediction not in prob_map or prob_map[prediction] < confidence:
            prob_map[prediction] = confidence

        for class_name, probability in prob_map.items():
            class_scores[class_name] = class_scores.get(class_name, 0.0) + probability

    n_models = len(successful)
    averaged = {cls: score / n_models for cls, score in class_scores.items()}
    final_prediction = max(averaged, key=averaged.get)
    final_confidence = averaged[final_prediction]

    vote_counts = Counter(votes)
    majority_prediction, majority_votes = vote_counts.most_common(1)[0]

    top_3 = sorted(
        [{'class': cls, 'probability': prob} for cls, prob in averaged.items()],
        key=lambda item: item['probability'],
        reverse=True,
    )[:3]

    return {
        'prediction': final_prediction,
        'confidence': final_confidence,
        'method': 'soft_voting',
        'models_used': list(successful.keys()),
        'model_count': n_models,
        'majority_vote': {
            'prediction': majority_prediction,
            'votes': majority_votes,
            'total': n_models,
        },
        'top_3': top_3,
    }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_names=model_names, targets=targets)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        complaint_text = data.get('text', '').strip()
        target = data.get('target', 'product')  # Default to product if not specified
        
        if not complaint_text:
            return jsonify({
                'success': False,
                'error': 'Please enter complaint text'
            }), 400
        
        if target not in targets:
            return jsonify({
                'success': False,
                'error': f'Invalid target. Must be one of: {", ".join(targets)}'
            }), 400
        
        has_tfidf = target in models and bool(models[target])
        has_bert = target in bert_models
        if not has_tfidf and not has_bert:
            return jsonify({
                'success': False,
                'error': f'No models loaded for {target}. Please train models first.'
            }), 500

        results = {}
        
        # Get predictions from all TF-IDF models for the specified target
        if target in models:
            for model_name, model in models[target].items():
                prediction, confidence, top_3 = predict_complaint(complaint_text, model, target)
                
                if prediction is not None:
                    results[model_name] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'top_3': top_3
                    }
                else:
                    results[model_name] = {
                        'error': 'Prediction failed'
                    }
        
        # Get prediction from BERT model for the specified target
        if target in bert_models:
            prediction, confidence, top_3 = predict_complaint_bert(complaint_text, target)
            
            if prediction is not None:
                results['BERT'] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'top_3': top_3
                }
            else:
                results['BERT'] = {
                    'error': 'BERT prediction failed'
                }

        final_ensemble_prediction = compute_ensemble_prediction(results)
        if not final_ensemble_prediction:
            return jsonify({
                'success': False,
                'error': 'All model predictions failed. Please try again.'
            }), 500

        return jsonify({
            'success': True,
            'target': target,
            'results': results,
            'final_ensemble_prediction': final_ensemble_prediction,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe uploaded audio with Groq Whisper."""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided. Use multipart field name "audio".'
            }), 400

        client = get_groq_client()
        if client is None:
            return jsonify({
                'success': False,
                'error': 'Voice transcription is not configured. Set GROQ_API_KEY in .env and restart the server.'
            }), 503

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({
                'success': False,
                'error': 'Empty audio upload.'
            }), 400
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return jsonify({
                'success': False,
                'error': 'Audio file too large (max 25 MB).'
            }), 400

        filename = audio_file.filename or 'recording.webm'
        mimetype = audio_file.mimetype or 'audio/webm'

        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes, mimetype),
            model=GROQ_WHISPER_MODEL,
            language='en',
            response_format='json',
            temperature=0.0,
        )
        text = (transcription.text or '').strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'No speech detected. Please try again.'
            }), 422

        return jsonify({
            'success': True,
            'text': text,
            'model': GROQ_WHISPER_MODEL,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Transcription failed: {e}'
        }), 500

@app.route('/models/status')
def models_status():
    """Get status of loaded models"""
    status = {
        'model_names': model_names,
        'targets': targets,
        'targets_status': {},
        'transcription': {
            'available': get_groq_client() is not None,
            'model': GROQ_WHISPER_MODEL,
        },
    }
    
    for target in targets:
        tfidf_models = list(models.get(target, {}).keys())
        bert_loaded = target in bert_models
        all_model_names = tfidf_models.copy()
        if bert_loaded:
            all_model_names.append('BERT')
        
        status['targets_status'][target] = {
            'models_loaded': len(models.get(target, {})) + (1 if bert_loaded else 0),
            'vectorizer_loaded': target in vectorizers and vectorizers[target] is not None,
            'label_encoder_loaded': target in label_encoders and label_encoders[target] is not None,
            'bert_loaded': bert_loaded,
            'model_names': all_model_names
        }
    
    return jsonify(status)

if __name__ == '__main__':
    print("=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    if load_models():
        total_models = sum(len(models[target]) for target in targets if target in models)
        print(f"\n[OK] Successfully loaded {total_models} model(s) across {len(targets)} target(s)")
        print(f"  Available models: {', '.join(model_names)}")
        print("\n" + "=" * 80)
        print("Starting Flask server...")
        print("Open http://127.0.0.1:5000 in your browser")
        print("=" * 80)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n[ERROR] Failed to load models. Please train models first using the notebook.")

