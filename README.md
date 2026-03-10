# 🧠 AI-Based Exam Anxiety Detector

An end-to-end NLP project that **detects exam anxiety levels** from student text using a fine-tuned **BERT model**, exposed via **FastAPI**, and consumed by a **Streamlit** frontend.

## 📊 Features

- **BERT-based Classification**: Uses `bert-base-uncased` for understanding contextual exam anxiety from text
- **FastAPI Backend**: RESTful API for making predictions
- **Streamlit Frontend**: User-friendly web interface for real-time anxiety detection
- **Confidence Scores**: Returns anxiety level with confidence percentage
- **Logging**: Tracks all predictions for monitoring and debugging

---

## 🎯 Anxiety Levels

The model classifies anxiety into **3 categories**:

| Level | Emoji | Indicator | Suggestion |
|-------|-------|-----------|-----------|
| **Low** | 😊 | Feeling confident | Keep up preparation and maintain confidence |
| **Moderate** | 😐 | Some nervousness | Take breaks and practice relaxation techniques |
| **High** | 😟 | Very anxious | Talk to counselor/teacher and practice breathing |

---

## 📁 Project Structure

```
AI_Exam_Anxiety_Detector/
├── data/
│   └── dataset.csv                 # Training dataset (auto-generated if missing)
├── model/
│   ├── train_model.py              # Training script
│   ├── anxiety_model.pt            # Trained model (generated)
│   └── anxiety_model_meta.json     # Model metadata (generated)
├── backend/
│   ├── main.py                     # FastAPI application
│   └── predictions.log             # Prediction logs (generated)
├── frontend/
│   └── app.py                      # Streamlit application
├── utils/
│   ├── preprocessing.py            # Text preprocessing utilities
│   └── predict.py                  # Prediction pipeline
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

---

## 🔧 How It Works

### BERT Model Architecture (Simple Explanation)

**BERT** (Bidirectional Encoder Representations from Transformers) reads text in both directions:

1. **Tokenization**: Text → BERT subword tokens
2. **Embedding**: Tokens → Contextual embeddings (understanding words with context)
3. **Classification Head**: Embeddings → 3 logits (Low, Moderate, High)
4. **Softmax**: Logits → Confidence probabilities
5. **Prediction**: Highest probability = predicted anxiety level

### Prediction Pipeline

```
User Input (Streamlit)
        ↓
POST /predict (FastAPI)
        ↓
Text Preprocessing
        ↓
BERT Tokenization
        ↓
Model Inference
        ↓
Softmax + Probabilities
        ↓
Response: {anxiety_level, confidence, scores}
        ↓
Display Results (Streamlit)
```

---

## 📊 Dataset Details

- **Location**: `data/dataset.csv`
- **Auto-generation**: If missing, `train_model.py` generates a synthetic dataset
- **Format**: `text,label`
- **Size**: 240 samples (configurable)
- **Labels**: `Low`, `Moderate`, `High`
- **Split**: 80% training, 20% validation (stratified)

### Example Dataset

```csv
text,label
I feel very prepared for my exam,Low
I'm a bit nervous about the upcoming test,Moderate
I can't sleep thinking about the exam,High
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)

### Step-by-Step Instructions

#### 1️⃣ Clone/Navigate to Project

```bash
cd C:\Smartbridge\AI_Exam_Anxiety_Detector
```

#### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

#### 3️⃣ Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

> **Note**: If you get an execution policy error on PowerShell, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies installed:
- `transformers` - BERT model
- `torch` - Deep learning framework
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `streamlit` - Frontend framework
- `numpy` - Numerical computing

#### 5️⃣ Train the Model (First Time Only)

```bash
python model/train_model.py
```

**Output:**
- ✅ Generates `data/dataset.csv` if missing
- ✅ Saves `model/anxiety_model.pt`
- ✅ Saves `model/anxiety_model_meta.json`
- 📊 Training logs and metrics displayed

**Training Configuration:**
- **Model**: `bert-base-uncased`
- **Optimizer**: AdamW (learning rate: 2e-5)
- **Loss Function**: Cross Entropy
- **Epochs**: 3
- **Batch Size**: 8-16 (varies per implementation)

---

## 🎮 Running the Application

### Option 1: Full Stack (Recommended)

#### Terminal 1 - Start Backend API

```bash
cd C:\Smartbridge\AI_Exam_Anxiety_Detector
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

#### Terminal 2 - Start Frontend

```bash
cd C:\Smartbridge\AI_Exam_Anxiety_Detector
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

**Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

#### Step 3 - Access the Application

1. Open browser and go to **`http://localhost:8501`**
2. Enter exam-related text in the text area
3. Click **"Predict"** button
4. View results with emoji, confidence, and suggestions

### Option 2: Backend Only (API Testing)

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then test with curl or Postman:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am very anxious about my upcoming exam\"}"
```

**Sample Response:**

```json
{
  "anxiety_level": "High",
  "confidence": 0.92,
  "scores": {
    "Low": 0.05,
    "Moderate": 0.03,
    "High": 0.92
  }
}
```

### Option 3: Frontend Only (Manual Testing)

If you only want to test the Streamlit UI:

```bash
streamlit run frontend/app.py
```

*(Note: Requires backend to be running for predictions)*

---

## 📝 API Documentation

### Endpoint: `/predict`

**Method**: `POST`

**Request Body**:
```json
{
  "text": "I am feeling anxious about my exam"
}
```

**Validation**:
- `text` must be between 3-1000 characters

**Response** (200 OK):
```json
{
  "anxiety_level": "Moderate",
  "confidence": 0.87,
  "scores": {
    "Low": 0.08,
    "Moderate": 0.87,
    "High": 0.05
  }
}
```

**Error Response** (422 Unprocessable Entity):
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at least 3 characters",
      "type": "value_error.str.min_length"
    }
  ]
}
```

---

## 📊 Logging

All predictions are logged to `backend/predictions.log`:

```
2026-03-10 14:35:42,123 | INFO | Text: "I feel nervous..." | Prediction: High | Confidence: 0.92
2026-03-10 14:36:15,456 | INFO | Text: "I'm well prepared..." | Prediction: Low | Confidence: 0.95
```

---

## 🛠️ Development & Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure virtual environment is activated:
```bash
.\.venv\Scripts\Activate.ps1
```

### Issue: Port 8000 already in use

**Solution**: Use a different port:
```bash
uvicorn backend.main:app --port 8001
```

### Issue: Model file not found

**Solution**: Train the model first:
```bash
python model/train_model.py
```

### Issue: CUDA/GPU errors

**Solution**: The code runs on CPU. If you need GPU, install `torch` with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | Latest | BERT model and utilities |
| torch | Latest | Deep learning framework |
| pandas | Latest | Data manipulation |
| scikit-learn | Latest | ML utilities |
| fastapi | Latest | Web API framework |
| uvicorn | Latest | ASGI server |
| streamlit | Latest | Frontend framework |
| numpy | Latest | Numerical computing |

---

## 🎓 Learning Resources

- **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Hugging Face Transformers**: [transformers.readthedocs.io](https://transformers.readthedocs.io/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **Streamlit**: [streamlit.io](https://streamlit.io/)

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 👨‍💻 Author

Created by Smartbridge

---

## 📞 Support

For issues or questions, refer to the troubleshooting section above or check the logs in `backend/predictions.log`.
