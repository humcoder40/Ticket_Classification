# Complaint Classification System - Model Architecture & Implementation Report

## Executive Summary

This document provides a comprehensive overview of the Complaint Classification System, detailing all machine learning models implemented, their theoretical foundations, implementation details, and the overall system architecture. Our system employs a **multi-model ensemble approach** where we train 4-5 different models and leverage their diverse strengths to achieve robust classification performance across varying scenarios.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Descriptions](#model-descriptions)
3. [Implementation Details](#implementation-details)
4. [System Architecture](#system-architecture)
5. [Ensemble Approach](#ensemble-approach)
6. [Data Pipeline](#data-pipeline)
7. [Performance Metrics](#performance-metrics)

---

## System Overview

### Purpose
The Complaint Classification System automatically categorizes consumer complaints into:
- **Product Categories** (e.g., Credit card, Mortgage, Bank account)
- **Issue Categories** (e.g., Loan modification, Managing an account, Billing disputes)

### Key Features
- **Multi-Model Ensemble**: Trains 4-5 different models per classification task
- **Dual Classification**: Separate models for Product and Issue classification
- **Confidence-Based Selection**: Displays all model predictions with confidence scores
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Predictions**: Fast inference for all models simultaneously

---

## Model Descriptions

### 1. Naive Bayes Classifier

#### What It Is
Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong independence assumptions between features. It's particularly effective for text classification tasks.

#### How It Works
1. **Training Phase**:
   - Calculates prior probabilities for each class: P(class)
   - Computes conditional probabilities: P(feature|class) for each feature
   - Uses Laplace smoothing (α=0.8) to handle zero probabilities

2. **Prediction Phase**:
   - For a given text, calculates: P(class|features) ∝ P(class) × ∏P(feature|class)
   - Selects the class with highest posterior probability

#### Mathematical Foundation
```
P(class|features) = P(class) × P(feature₁|class) × P(feature₂|class) × ... × P(featureₙ|class)
```

#### Strengths
- Fast training and prediction
- Works well with high-dimensional sparse data (like TF-IDF vectors)
- Handles missing features gracefully
- Good baseline model

#### Weaknesses
- Assumes feature independence (often violated in text)
- Can be sensitive to irrelevant features
- May struggle with complex relationships

#### Implementation in Our System
- **Library**: `sklearn.naive_bayes.MultinomialNB`
- **Parameters**: `alpha=0.8` (Laplace smoothing)
- **Input**: TF-IDF vectorized text (80,000 features)
- **Output**: Class probabilities and predictions

---

### 2. Logistic Regression

#### What It Is
Logistic Regression is a linear classification algorithm that uses the logistic function to model the probability of a class. It's a discriminative model that learns decision boundaries.

#### How It Works
1. **Training Phase**:
   - Learns weights (coefficients) for each feature
   - Uses gradient descent to minimize cross-entropy loss
   - Applies L2 regularization (C=2.0) to prevent overfitting
   - Uses class weighting ('balanced') to handle imbalanced data

2. **Prediction Phase**:
   - Computes: P(class|features) = 1 / (1 + e^(-z))
   - Where z = w₀ + w₁×feature₁ + w₂×feature₂ + ... + wₙ×featureₙ
   - Selects class with highest probability

#### Mathematical Foundation
```
P(y=1|x) = 1 / (1 + e^(-(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)))
```

#### Strengths
- Interpretable (can see feature importance)
- Fast training and prediction
- Works well with sparse data
- Provides probability estimates
- Handles imbalanced data with class weighting

#### Weaknesses
- Assumes linear decision boundaries
- May struggle with non-linear relationships
- Requires feature scaling (handled by TF-IDF)

#### Implementation in Our System
- **Library**: `sklearn.linear_model.LogisticRegression`
- **Parameters**:
  - `C=2.0` (inverse regularization strength)
  - `class_weight='balanced'` (handles imbalanced classes)
  - `max_iter=1500` (maximum iterations)
  - `solver='liblinear'` (optimization algorithm)
- **Input**: TF-IDF vectorized text
- **Output**: Class probabilities and predictions

---

### 3. Random Forest Classifier

#### What It Is
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes (classification) from individual trees.

#### How It Works
1. **Training Phase**:
   - Creates 300 decision trees (n_estimators=300)
   - Each tree is trained on a bootstrap sample of data
   - At each split, considers random subset of features
   - Trees grow to full depth (max_depth=None)

2. **Prediction Phase**:
   - Each tree makes a prediction
   - Final prediction is majority vote across all trees
   - Probabilities are averaged across trees

#### Mathematical Foundation
```
Prediction = Mode({Tree₁(x), Tree₂(x), ..., Treeₙ(x)})
Probability = Average({P₁(x), P₂(x), ..., Pₙ(x)})
```

#### Strengths
- Handles non-linear relationships
- Reduces overfitting through ensemble
- Can capture feature interactions
- Provides feature importance scores
- Robust to outliers

#### Weaknesses
- Slower than linear models
- Less interpretable than single decision tree
- Can overfit with too many trees
- Memory intensive

#### Implementation in Our System
- **Library**: `sklearn.ensemble.RandomForestClassifier`
- **Parameters**:
  - `n_estimators=300` (number of trees)
  - `max_depth=None` (unlimited depth)
  - `min_samples_split=2` (minimum samples to split)
  - `min_samples_leaf=1` (minimum samples in leaf)
- **Input**: TF-IDF vectorized text
- **Output**: Class probabilities and predictions

---

### 4. Support Vector Machine (SVM)

#### What It Is
SVM finds the optimal hyperplane that separates classes with maximum margin. For text classification, linear SVMs are particularly effective.

#### How It Works
1. **Training Phase**:
   - Finds optimal separating hyperplane
   - Maximizes margin between classes
   - Uses linear kernel (efficient for high-dimensional sparse data)
   - Applies regularization (C=2.0)
   - Uses class weighting ('balanced') for imbalanced data

2. **Prediction Phase**:
   - Computes distance to hyperplane
   - Converts to probability using Platt scaling
   - Selects class with highest probability

#### Mathematical Foundation
```
Decision Function: f(x) = wᵀx + b
Prediction: sign(f(x))
```

#### Strengths
- Effective with high-dimensional data
- Memory efficient (uses support vectors only)
- Works well with sparse data (TF-IDF)
- Good generalization
- Often achieves high accuracy

#### Weaknesses
- Slower training than linear models
- Less interpretable
- Requires probability calibration for probabilities
- Sensitive to feature scaling (handled by TF-IDF)

#### Implementation in Our System
- **Library**: `sklearn.svm.SVC`
- **Parameters**:
  - `kernel='linear'` (linear kernel for efficiency)
  - `C=2.0` (regularization parameter)
  - `class_weight='balanced'` (handles imbalanced classes)
  - `probability=True` (enables probability estimates)
- **Input**: TF-IDF vectorized text
- **Output**: Class probabilities and predictions

---

### 5. BERT (Bidirectional Encoder Representations from Transformers)

#### What It Is
BERT is a transformer-based deep learning model pre-trained on large text corpora. We use DistilBERT, a lighter, faster version that retains most of BERT's performance.

#### How It Works
1. **Pre-training** (Done by HuggingFace):
   - Trained on Wikipedia and BookCorpus
   - Learns bidirectional context understanding
   - Understands word relationships and context

2. **Fine-tuning** (Our Implementation):
   - Takes pre-trained DistilBERT model
   - Adds classification head (linear layer)
   - Fine-tunes on complaint classification task
   - Learns task-specific representations

3. **Prediction Phase**:
   - Tokenizes input text (max 256 tokens)
   - Passes through transformer layers
   - Generates contextual embeddings
   - Classification head outputs class probabilities

#### Mathematical Foundation
```
BERT(x) = Transformer(Embedding(x))
P(class|x) = Softmax(Linear(BERT(x)))
```

#### Strengths
- Understands context and semantics
- Captures long-range dependencies
- State-of-the-art performance
- Handles complex language patterns
- Pre-trained on vast text data

#### Weaknesses
- Requires significant computational resources
- Slower inference than traditional ML models
- Requires GPU for efficient training
- Larger model size

#### Implementation in Our System
- **Library**: `transformers.AutoModelForSequenceClassification`
- **Base Model**: `distilbert-base-uncased`
- **Parameters**:
  - `max_length=256` (sequence length)
  - `batch_size=8` (training)
  - `learning_rate=2e-5` (fine-tuning)
  - `epochs=1` (fine-tuning epochs)
- **Input**: Raw text (tokenized internally)
- **Output**: Class probabilities and predictions
- **Device**: CUDA GPU if available, else CPU

---

## Implementation Details

### Data Preprocessing Pipeline

```
Raw Complaint Text
    ↓
Text Cleaning (remove extra whitespace)
    ↓
Preprocessed Text
    ↓
┌─────────────────────┬─────────────────────┐
│   TF-IDF Path       │    BERT Path        │
│                     │                      │
│ TF-IDF Vectorization│  Tokenization       │
│ (80,000 features)   │  (Max 256 tokens)    │
│                     │                      │
│ Feature Matrix      │  Token IDs +         │
│ (sparse matrix)     │  Attention Mask      │
└─────────────────────┴─────────────────────┘
```

### Model Training Flow

1. **Data Loading**:
   - Load JSON complaint data (78,313 records)
   - Extract `complaint_what_happened` (text) and target labels

2. **Data Cleaning**:
   - Remove missing values
   - Filter empty strings
   - Filter to top 10 classes per target

3. **Text Preprocessing**:
   - Normalize whitespace
   - Clean text (basic cleaning)

4. **Feature Extraction**:
   - **TF-IDF Models**: Convert text to TF-IDF vectors (80,000 features, 1-2 grams)
   - **BERT**: Tokenize text (max 256 tokens)

5. **Model Training**:
   - Split data (80% train, 20% test)
   - Train each model separately
   - Evaluate on test set

6. **Model Saving**:
   - Save models as pickle files (TF-IDF models)
   - Save BERT models in HuggingFace format

### Prediction Flow

```
User Input (Complaint Text)
    ↓
Text Preprocessing
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Parallel Processing                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  TF-IDF Models:          │  BERT Model:                │
│  1. Naive Bayes          │  1. Tokenize                │
│  2. Logistic Regression  │  2. Pass through BERT       │
│  3. Random Forest        │  3. Get logits              │
│  4. SVM                  │  4. Apply softmax           │
│                          │                              │
│  Each model:             │  Output:                     │
│  - Vectorize text        │  - Prediction                │
│  - Predict class         │  - Confidence                │
│  - Get probabilities     │  - Top 3 predictions         │
│                          │                              │
└─────────────────────────────────────────────────────────┘
    ↓
Collect All Predictions
    ↓
Display Results with Confidence Scores
```

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEB INTERFACE (Flask)                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Frontend (HTML/CSS/JS)                        │  │
│  │  - User Input Form                                               │  │
│  │  - Model Selection (Product/Issue)                               │  │
│  │  - Results Display                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/JSON
┌─────────────────────────────────────────────────────────────────────────┐
│                      FLASK APPLICATION SERVER                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    API Endpoints                                 │  │
│  │  - / (Home)                                                      │  │
│  │  - /predict (POST)                                               │  │
│  │  - /models/status (GET)                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL LOADING LAYER                                 │
│  ┌──────────────────────┐         ┌──────────────────────┐            │
│  │   TF-IDF Models      │         │    BERT Models       │            │
│  │                      │         │                      │            │
│  │  - Naive Bayes       │         │  - Product BERT      │            │
│  │  - Logistic Reg.     │         │  - Issue BERT        │            │
│  │  - Random Forest     │         │                      │            │
│  │  - SVM               │         │  + Tokenizers        │            │
│  │                      │         │  + Label Encoders     │            │
│  │  + Vectorizers       │         │                      │            │
│  │  + Label Encoders    │         │                      │            │
│  └──────────────────────┘         └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREDICTION ENGINE                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Text Preprocessing                            │  │
│  │  - Clean text                                                    │  │
│  │  - Normalize whitespace                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↕                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Parallel Model Execution                            │  │
│  │                                                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │  │
│  │  │ Naive Bayes   │  │ Log. Reg.     │  │ Random Forest │       │  │
│  │  │ Prediction    │  │ Prediction    │  │ Prediction     │       │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │  │
│  │                                                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐                           │  │
│  │  │ SVM           │  │ BERT          │                           │  │
│  │  │ Prediction    │  │ Prediction    │                           │  │
│  │  └──────────────┘  └──────────────┘                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↕                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Result Aggregation                            │  │
│  │  - Collect all predictions                                       │  │
│  │  - Calculate confidence scores                                    │  │
│  │  - Get top 3 predictions per model                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESPONSE FORMATTING                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  JSON Response:                                                  │  │
│  │  {                                                               │  │
│  │    "success": true,                                             │  │
│  │    "target": "product",                                         │  │
│  │    "results": {                                                 │  │
│  │      "Naive Bayes": {...},                                      │  │
│  │      "Logistic Regression": {...},                              │  │
│  │      "Random Forest": {...},                                   │  │
│  │      "SVM": {...},                                              │  │
│  │      "BERT": {...}                                              │  │
│  │    }                                                             │  │
│  │  }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Component Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         START: User Request                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: Receive Complaint Text + Target (Product/Issue)             │
│  - Validate input                                                    │
│  - Check target validity                                             │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 2: Text Preprocessing                                         │
│  - Remove extra whitespace                                           │
│  - Clean text                                                        │
│  - Normalize format                                                  │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 3: Feature Extraction (Parallel)                               │
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────┐     │
│  │  TF-IDF Vectorization    │    │  BERT Tokenization       │     │
│  │  - Transform text        │    │  - Tokenize text         │     │
│  │  - Create sparse vector  │    │  - Create token IDs      │     │
│  │  - 80,000 features       │    │  - Attention mask         │     │
│  └──────────────────────────┘    └──────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 4: Model Predictions (Parallel Execution)                     │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Naive Bayes │  │ Log. Reg.   │  │ Rand. Forest│  │   SVM    │ │
│  │             │  │             │  │             │  │          │ │
│  │ Input:      │  │ Input:      │  │ Input:      │  │ Input:   │ │
│  │ TF-IDF vec  │  │ TF-IDF vec  │  │ TF-IDF vec  │  │ TF-IDF  │ │
│  │             │  │             │  │             │  │ vec      │ │
│  │ Output:     │  │ Output:     │  │ Output:     │  │ Output:  │ │
│  │ - Class     │  │ - Class     │  │ - Class     │  │ - Class  │ │
│  │ - Conf.     │  │ - Conf.     │  │ - Conf.     │  │ - Conf.  │ │
│  │ - Top 3     │  │ - Top 3     │  │ - Top 3     │  │ - Top 3  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    BERT Model                                  │ │
│  │  Input: Token IDs + Attention Mask                             │ │
│  │  Process:                                                      │ │
│  │  1. Pass through transformer layers                            │ │
│  │  2. Get contextual embeddings                                  │ │
│  │  3. Classification head                                       │ │
│  │  4. Softmax for probabilities                                 │ │
│  │  Output:                                                       │ │
│  │  - Class                                                        │ │
│  │  - Confidence                                                  │ │
│  │  - Top 3 predictions                                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 5: Result Aggregation                                        │
│  - Collect predictions from all models                              │
│  - Format results                                                   │
│  - Include confidence scores                                         │
│  - Include top 3 predictions per model                               │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 6: Return JSON Response                                      │
│  - Send to frontend                                                 │
│  - Display in UI                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Ensemble Approach

### Implemented Final Prediction (Production)

The live Flask app aggregates all successful model outputs into a single **`final_ensemble_prediction`** returned by `POST /predict` and shown prominently in the web UI.

**Method: Soft voting**

1. Each successful model contributes class scores from its `top_3` probabilities (the predicted class uses at least its reported confidence).
2. Scores are averaged across all models that returned a valid prediction.
3. The class with the highest average score becomes the **Final Ensemble Prediction**.
4. **Majority vote** metadata is included for transparency (`votes` / `total`).

**Models in the ensemble (when loaded):**

| Model | Path |
|-------|------|
| Naive Bayes | TF-IDF + `models/{target}_naive_bayes_model.pkl` |
| Logistic Regression | TF-IDF |
| Random Forest | TF-IDF |
| SVM | TF-IDF |
| BERT | `bert_model_{target}/` (DistilBERT fine-tuned) |

**Example API field:**

```json
"final_ensemble_prediction": {
  "prediction": "Credit card or prepaid card",
  "confidence": 0.601,
  "method": "soft_voting",
  "models_used": ["Naive Bayes", "Logistic Regression", "Random Forest", "SVM", "BERT"],
  "model_count": 5,
  "majority_vote": { "prediction": "Credit card or prepaid card", "votes": 4, "total": 5 },
  "top_3": [...]
}
```

### Our Multi-Model Strategy

**Why We Train Multiple Models:**

Different machine learning models excel in different scenarios due to their inherent characteristics:

1. **Naive Bayes**: Fast, probabilistic, good baseline
2. **Logistic Regression**: Linear, interpretable, handles imbalanced data
3. **Random Forest**: Non-linear, robust, captures interactions
4. **SVM**: High accuracy, good with sparse data
5. **BERT**: Context-aware, state-of-the-art, understands semantics

### Display Strategy

1. **Train Multiple Models**: Each model learns different patterns
2. **Run All Models**: Every prediction uses all available models
3. **Aggregate with Soft Voting**: One final label + confidence for decision-making
4. **Show Individual Results**: Users still see each model's prediction and top-3 scores
5. **Show Majority Vote**: Highlights agreement vs. disagreement across models

### Why This Approach Works

#### Scenario 1: Simple, Clear Text
- **Naive Bayes** or **Logistic Regression** might be highly confident
- Fast, reliable predictions

#### Scenario 2: Complex Relationships
- **Random Forest** captures non-linear patterns
- Higher confidence for complex cases

#### Scenario 3: High-Dimensional Sparse Data
- **SVM** excels with TF-IDF vectors
- Often highest accuracy

#### Scenario 4: Context-Dependent Text
- **BERT** understands semantics and context
- Best for nuanced language

### Benefits of Our Approach

1. **Robustness**: If one model fails, others provide backup
2. **Transparency**: Users see all predictions, not just one
3. **Flexibility**: Different models for different scenarios
4. **Confidence Assessment**: Multiple confidence scores help assess reliability
5. **No Single Point of Failure**: Ensemble reduces risk

### Example Scenario

**Input**: "I have been trying to resolve an issue with my credit card statement. The charges are incorrect and I have been charged fees that I should not have been charged."

**Individual model predictions** (illustrative):

- **Naive Bayes**: Credit card or prepaid card (75.5%)
- **Logistic Regression**: Credit card or prepaid card (60.5%)
- **Random Forest**: Credit card or prepaid card (45.0%)
- **SVM**: Credit card or prepaid card (80.5%)
- **BERT**: Credit card (57.2%)

**Final ensemble (soft voting)**: Credit card or prepaid card (~60%)  
**Majority vote**: 4/5 models agree on "Credit card or prepaid card"

---

## Data Pipeline

### Training Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (78,313 complaints)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA CLEANING                                │
│  - Remove missing values                                        │
│  - Filter empty strings                                         │
│  - Result: ~21,000 clean records                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CLASS FILTERING                              │
│  - Filter to top 10 classes per target                         │
│  - Product: 20,777 records                                     │
│  - Issue: 9,672 records                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT PREPROCESSING                           │
│  - Normalize whitespace                                         │
│  - Basic cleaning                                               │
│  - Create 'text_processed' column                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
│                                                                 │
│  ┌────────────────────────┐    ┌────────────────────────┐   │
│  │  TF-IDF Vectorization  │    │  BERT Preparation       │   │
│  │                        │    │                        │   │
│  │  - Max features: 80K   │    │  - Max length: 256     │   │
│  │  - N-grams: (1,2)      │    │  - Sample: 500/class   │   │
│  │  - Min DF: 2           │    │  - Tokenization        │   │
│  │  - Max DF: 0.98        │    │                        │   │
│  │  - Sublinear TF: True  │    │                        │   │
│  └────────────────────────┘    └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING                               │
│  - Train: 80%                                                   │
│  - Test: 20%                                                    │
│  - Stratified split                                             │
│  - Random state: 42                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Naive Bayes  │  │ Log. Reg.    │  │ Rand. Forest │        │
│  │ Training     │  │ Training     │  │ Training     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ SVM          │  │ BERT         │                           │
│  │ Training     │  │ Fine-tuning  │                           │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│  - Test set predictions                                         │
│  - Accuracy scores                                              │
│  - Classification reports                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SAVING                                 │
│  - TF-IDF models: Pickle files                                  │
│  - BERT models: HuggingFace format                             │
│  - Vectorizers: Pickle files                                    │
│  - Label encoders: Pickle files                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                    │
│  - Complaint text                                               │
│  - Target selection (Product/Issue)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT PREPROCESSING                           │
│  - Clean text                                                    │
│  - Normalize whitespace                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
│                                                                 │
│  ┌────────────────────────┐    ┌────────────────────────┐   │
│  │  TF-IDF Vectorization  │    │  BERT Tokenization     │   │
│  │  - Transform text      │    │  - Tokenize            │   │
│  │  - Sparse vector       │    │  - Create IDs          │   │
│  └────────────────────────┘    └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL PREDICTIONS                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Naive Bayes  │  │ Log. Reg.    │  │ Rand. Forest │        │
│  │ Prediction   │  │ Prediction   │  │ Prediction   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ SVM          │  │ BERT        │                           │
│  │ Prediction   │  │ Prediction  │                           │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT AGGREGATION                           │
│  - Collect all predictions                                      │
│  - Calculate confidence scores                                  │
│  - Get top 3 per model                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    JSON RESPONSE                                 │
│  {                                                               │
│    "success": true,                                              │
│    "target": "product",                                          │
│    "results": {                                                  │
│      "Naive Bayes": {...},                                       │
│      "Logistic Regression": {...},                                │
│      "Random Forest": {...},                                     │
│      "SVM": {...},                                               │
│      "BERT": {...}                                               │
│    }                                                              │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Metrics

### Model Performance (Typical Results)

#### Product Classification
- **Naive Bayes**: ~59-60% accuracy
- **Logistic Regression**: ~70-71% accuracy
- **Random Forest**: ~65-66% accuracy
- **SVM**: ~71-72% accuracy (often highest)
- **BERT**: ~75-80% accuracy (with proper fine-tuning)

#### Issue Classification
- **Naive Bayes**: ~51-52% accuracy
- **Logistic Regression**: ~69-70% accuracy
- **Random Forest**: ~65-66% accuracy
- **SVM**: ~71-72% accuracy
- **BERT**: ~75-80% accuracy (with proper fine-tuning)

### Why Different Models Perform Differently

1. **Data Characteristics**: Different models suit different data distributions
2. **Feature Relationships**: Linear vs non-linear patterns
3. **Class Imbalance**: Some models handle imbalance better
4. **Text Complexity**: Simple vs complex language patterns
5. **Context Understanding**: BERT excels at semantic understanding

---

## Technical Specifications

### System Requirements

- **Python**: 3.7+
- **Libraries**:
  - Flask (Web framework)
  - scikit-learn (ML models)
  - transformers (BERT)
  - torch (Deep learning)
  - pandas, numpy (Data processing)

### Model Storage

- **TF-IDF Models**: `models/` directory
  - `{target}_{model_name}_model.pkl`
  - `{target}_tfidf_vectorizer.pkl`
  - `{target}_label_encoder.pkl`

- **BERT Models**: Separate directories
  - `bert_model_product/`
  - `bert_model_issue/`

### API Endpoints

1. **GET /** - Home page
2. **POST /predict** - Get predictions
3. **GET /models/status** - Model status

---

## Conclusion

Our Complaint Classification System employs a sophisticated multi-model ensemble approach that leverages the strengths of different machine learning algorithms. By training 4-5 models and displaying all predictions with confidence scores, we provide:

1. **Robustness**: Multiple models reduce single-point-of-failure risk
2. **Transparency**: Users see all model predictions
3. **Flexibility**: Different models excel in different scenarios
4. **Confidence Assessment**: Multiple confidence scores aid decision-making
5. **State-of-the-Art Performance**: BERT provides cutting-edge accuracy

This approach recognizes that no single model is perfect for all scenarios, and by combining traditional ML models with modern deep learning, we achieve comprehensive coverage of text classification challenges.

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- HuggingFace Transformers: https://huggingface.co/transformers/
- BERT Paper: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- Flask Documentation: https://flask.palletsprojects.com/

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Author**: Complaint Classification System Team

