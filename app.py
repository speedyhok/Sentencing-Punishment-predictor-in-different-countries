# ==============================================================================
# Global Sentencing Proportionality Drift Auditor
# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <hokworks@gmail.com> (github.com/speedyhok | linkedin.com/in/speedymohibul)
# Description: Modular machine learning pipeline for cross-jurisdiction legal auditing.
# ==============================================================================

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Global variables for models
tokenizer = None
transformer_model = None
severity_model = None
country_models = None
country_residuals_std = None
typology_metadata = None

# List of 20 countries
COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Australia", "Singapore",
    "India", "Germany", "France", "Italy", "Spain", "Switzerland", "Brazil",
    "Norway", "Sweden", "Denmark", "Japan", "South Korea", "Saudi Arabia",
    "Iran", "South Africa"
]

# Country details
COUNTRY_DETAILS = {
    "United States": {"legal_family": "Common Law", "region": "North America"},
    "United Kingdom": {"legal_family": "Common Law", "region": "Europe"},
    "Canada": {"legal_family": "Common Law", "region": "North America"},
    "Australia": {"legal_family": "Common Law", "region": "Oceania"},
    "Singapore": {"legal_family": "Common Law / Mixed", "region": "Asia"},
    "India": {"legal_family": "Common Law / Mixed", "region": "Asia"},
    "Germany": {"legal_family": "Civil Law", "region": "Europe"},
    "France": {"legal_family": "Civil Law", "region": "Europe"},
    "Italy": {"legal_family": "Civil Law", "region": "Europe"},
    "Spain": {"legal_family": "Civil Law", "region": "Europe"},
    "Switzerland": {"legal_family": "Civil Law", "region": "Europe"},
    "Brazil": {"legal_family": "Civil Law", "region": "South America"},
    "Norway": {"legal_family": "Nordic Civil Law", "region": "Europe"},
    "Sweden": {"legal_family": "Nordic Civil Law", "region": "Europe"},
    "Denmark": {"legal_family": "Nordic Civil Law", "region": "Europe"},
    "Japan": {"legal_family": "Civil Law / East Asian", "region": "Asia"},
    "South Korea": {"legal_family": "Civil Law / East Asian", "region": "Asia"},
    "Saudi Arabia": {"legal_family": "Islamic Law", "region": "Middle East"},
    "Iran": {"legal_family": "Islamic Law", "region": "Middle East"},
    "South Africa": {"legal_family": "Mixed Law", "region": "Africa"}
}

def init_models():
    global tokenizer, transformer_model, severity_model, country_models, country_residuals_std, typology_metadata
    
    model_dir = "models"
    if not os.path.exists(os.path.join(model_dir, "severity_model.pkl")):
        raise FileNotFoundError("Models have not been trained yet. Please run the training pipeline first.")
        
    print("Loading severity mapping model and country models...")
    severity_model = joblib.load(os.path.join(model_dir, "severity_model.pkl"))
    country_models = joblib.load(os.path.join(model_dir, "country_models.pkl"))
    country_residuals_std = joblib.load(os.path.join(model_dir, "country_residuals_std.pkl"))
    typology_metadata = joblib.load(os.path.join(model_dir, "typology_metadata.pkl"))
    
    print("Loading HuggingFace transformer for online text embedding...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    transformer_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    transformer_model.eval()

def get_text_embedding(text):
    encoded = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        output = transformer_model(**encoded)
        
    token_embeddings = output[0]
    attention_mask = encoded["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).numpy()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/typology", methods=["GET"])
def get_typology():
    if typology_metadata is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    response_data = []
    for country, meta in typology_metadata["typology"].items():
        details = COUNTRY_DETAILS.get(country, {"legal_family": "Unknown", "region": "Unknown"})
        response_data.append({
            "country": country,
            "legal_family": details["legal_family"],
            "region": details["region"],
            "x": meta["x"],
            "y": meta["y"],
            "cluster_id": meta["cluster_id"],
            "cluster_name": meta["cluster_name"],
            "cluster_description": meta["cluster_description"]
        })
        
    return jsonify(response_data)

@app.route("/api/predict", methods=["POST"])
def predict():
    if severity_model is None:
        return jsonify({"error": "Models not loaded"}), 500
        
    data = request.get_json() or {}
    fact_pattern = data.get("fact_pattern", "")
    
    if not fact_pattern or len(fact_pattern.strip()) < 10:
        return jsonify({"error": "A case fact pattern of at least 10 characters is required."}), 400
        
    priors = int(data.get("priors", 0))
    plea_guilty = 1 if data.get("plea_guilty", False) else 0
    mitigating_circumstances = 1 if data.get("mitigating_circumstances", False) else 0
    juvenile = 1 if data.get("juvenile", False) else 0
    court_region = 1 if data.get("court_region", False) else 0 # 0 = Rural, 1 = Metro
    
    # Optional actual sentence
    actual_sentence = data.get("actual_sentence", None)
    if actual_sentence is not None:
        try:
            actual_sentence = float(actual_sentence)
        except ValueError:
            actual_sentence = None

    # Step 1: Text Embedding
    embedding = get_text_embedding(fact_pattern)
    
    # Step 2: Extract Severity Vector (5 dimensions)
    severity_vector = severity_model.predict(embedding)[0]
    severity_vector = np.clip(severity_vector, 0.0, 1.0)
    
    severity_scores = {
        "violence": round(float(severity_vector[0]), 3),
        "financial_loss": round(float(severity_vector[1]), 3),
        "victim_vulnerability": round(float(severity_vector[2]), 3),
        "premeditation": round(float(severity_vector[3]), 3),
        "public_safety_risk": round(float(severity_vector[4]), 3)
    }
    
    # Step 3: Run Country Predictions
    # Feature vector: 5 severity scores + 5 legal/case factors
    features = np.array([[
        severity_vector[0], severity_vector[1], severity_vector[2], 
        severity_vector[3], severity_vector[4], priors, plea_guilty, 
        mitigating_circumstances, juvenile, court_region
    ]])
    
    predictions = {}
    predicted_sentences_list = []
    
    for country in COUNTRIES:
        model = country_models[country]
        pred_val = float(model.predict(features)[0])
        pred_val = max(0.0, pred_val)
        predictions[country] = {
            "sentence": round(pred_val, 1)
        }
        predicted_sentences_list.append(pred_val)
        
    # Calculate reference median (20 jurisdictions) instead of global median
    reference_median = np.median(predicted_sentences_list)
    
    # Step 4: Layer in Proportionality Drift Calculations
    for country in COUNTRIES:
        pred_val = predictions[country]["sentence"]
        
        # Cross-Jurisdiction Proportionality Drift:
        # Zero-guarded percent deviation: (sentence - reference_median) / reference_median
        denominator = max(reference_median, 1e-6)
        cross_drift = (pred_val - reference_median) / denominator
        predictions[country]["cross_drift"] = round(cross_drift * 100, 1)
        
        # Typology details
        meta = typology_metadata["typology"][country]
        predictions[country]["cluster_id"] = meta["cluster_id"]
        predictions[country]["cluster_name"] = meta["cluster_name"]
        
        # Legal family details
        details = COUNTRY_DETAILS.get(country, {"legal_family": "Unknown", "region": "Unknown"})
        predictions[country]["legal_family"] = details["legal_family"]
        predictions[country]["region"] = details["region"]

        # Within-Jurisdiction Consistency Audit: Empirical Quantile Thresholds
        if actual_sentence is not None:
            residual = abs(actual_sentence - pred_val)
            bounds = country_residuals_std.get(country, {"q90": 10.0, "q95": 20.0})
            
            if residual <= bounds["q90"]:
                status = "Consistent"
            elif residual <= bounds["q95"]:
                status = "Moderate Drift"
            else:
                status = "Critical Drift (Anomaly)"
                
            predictions[country]["within_drift_diff"] = round(actual_sentence - pred_val, 1)
            predictions[country]["within_drift_status"] = status
            predictions[country]["threshold_90"] = round(bounds["q90"], 1)
            predictions[country]["threshold_95"] = round(bounds["q95"], 1)
        else:
            predictions[country]["within_drift_diff"] = None
            predictions[country]["within_drift_status"] = "No Actual Sentence Provided"
            
    return jsonify({
        "severity_scores": severity_scores,
        "reference_median_months": round(float(reference_median), 1),
        "predictions": predictions,
        "embedding_snippet": [round(float(x), 5) for x in embedding[0][:8]]
    })

if __name__ == "__main__":
    try:
        init_models()
    except Exception as e:
        print(f"Warning: Models could not be loaded on startup: {e}")
        
    app.run(host="0.0.0.0", port=5000, debug=True)
