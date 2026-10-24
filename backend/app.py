"""
AI Data Scientist Agent — FastAPI Backend
Now powered by Claude (Anthropic) instead of HuggingFace stubs.
"""
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import io

from data_analysis import DataAnalyzer
from ai_agent import ask_claude, ask_claude_about_dataset
from ml_model import MLModel
from report_generator import ReportGenerator
from database import SessionLocal, ChatHistory

app = FastAPI(title="AI Data Scientist Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state (one dataset + one model at a time)
uploaded_df = None
ml_instance = None


# ────────────────────────────────────────────
# ROOT
# ────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "AI Data Scientist Agent Backend Running ✅"}


# ────────────────────────────────────────────
# UPLOAD DATASET
# ────────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global uploaded_df, ml_instance
    ml_instance = None  # reset model when new data arrives

    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "csv":
            try:
                uploaded_df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        elif ext in ("xlsx", "xls"):
            uploaded_df = pd.read_excel(io.BytesIO(content))
        elif ext == "json":
            uploaded_df = pd.read_json(io.BytesIO(content))
        else:
            return {"error": f"Unsupported file type: {ext}. Use CSV, Excel, or JSON."}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    analyzer = DataAnalyzer(uploaded_df)
    return {
        "filename": file.filename,
        "message": "Upload Successful ✅",
        "dataset_info": analyzer.dataset_info()
    }


# ────────────────────────────────────────────
# DATA ANALYSIS
# ────────────────────────────────────────────
@app.get("/analysis")
def analysis():
    if uploaded_df is None:
        return {"error": "Upload a dataset first."}

    analyzer = DataAnalyzer(uploaded_df)
    return {
        "dataset_info":        analyzer.dataset_info(),
        "missing_values":      analyzer.missing_values(),
        "summary":             analyzer.summary_statistics(),
        "duplicates":          analyzer.duplicate_rows(),
        "data_types":          analyzer.data_types(),
        "cleaning_suggestions": analyzer.cleaning_suggestions(),
    }


# ────────────────────────────────────────────
# AI CHAT — now uses Claude
# ────────────────────────────────────────────
@app.post("/ask")
def ask_ai(question: str = Query(..., description="Your data science question")):
    """
    Ask Claude a question. If a dataset is loaded, Claude will
    receive its schema and stats as context for a smarter answer.
    """
    if uploaded_df is not None:
        answer = ask_claude_about_dataset(question, uploaded_df)
    else:
        answer = ask_claude(question)

    # Save to DB
    db = SessionLocal()
    try:
        chat = ChatHistory(question=question, answer=answer)
        db.add(chat)
        db.commit()
    finally:
        db.close()

    return {"question": question, "answer": answer}


@app.get("/chat-history")
def chat_history():
    db = SessionLocal()
    try:
        chats = db.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(50).all()
        return [
            {"id": c.id, "question": c.question, "answer": c.answer,
             "created_at": str(c.created_at)}
            for c in chats
        ]
    finally:
        db.close()


# ────────────────────────────────────────────
# TRAIN MODEL
# ────────────────────────────────────────────
@app.post("/train")
def train_model(target_column: str = Query(..., description="Column to predict")):
    global ml_instance

    if uploaded_df is None:
        return {"error": "Upload a dataset first."}
    if target_column not in uploaded_df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {list(uploaded_df.columns)}"}

    ml_instance = MLModel(uploaded_df)
    try:
        accuracy = ml_instance.train(target_column)
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

    return {
        "message": "Model trained successfully ✅",
        "target_column": target_column,
        "accuracy": round(accuracy * 100, 2)
    }


@app.get("/feature-importance")
def feature_importance():
    if ml_instance is None:
        return {"error": "Train a model first."}
    return ml_instance.get_feature_importance()


# ────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────
@app.post("/predict")
def predict(text: str = Query(..., description="Input value to predict")):
    if ml_instance is None:
        return {"error": "Train a model first."}
    try:
        prediction = ml_instance.predict(text)
        return {"input": text, "prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────
# DASHBOARD
# ────────────────────────────────────────────
@app.get("/dashboard")
def dashboard():
    if uploaded_df is None:
        return {"error": "Upload a dataset first."}
    return {
        "rows":     len(uploaded_df),
        "columns":  len(uploaded_df.columns),
        "column_names": list(uploaded_df.columns),
        "accuracy": round(ml_instance.get_accuracy() * 100, 2) if ml_instance else None,
    }


# ────────────────────────────────────────────
# PDF REPORT
# ────────────────────────────────────────────
@app.get("/generate-report")
def generate_report():
    if uploaded_df is None:
        return {"error": "Upload a dataset first."}
    if ml_instance is None:
        return {"error": "Train a model first."}

    analysis = {"rows": len(uploaded_df), "columns": len(uploaded_df.columns)}
    fi = ml_instance.get_feature_importance()

    pdf_file = ReportGenerator.generate_report(
        analysis,
        ml_instance.get_accuracy() * 100,
        fi
    )
    return FileResponse(path=pdf_file, media_type="application/pdf", filename="AI_Report.pdf")