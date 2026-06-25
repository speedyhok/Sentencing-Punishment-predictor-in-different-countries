# ==============================================================================
# Global Sentencing Proportionality Drift Auditor
# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <hokworks@gmail.com> (github.com/speedyhok | linkedin.com/in/speedymohibul)
# Description: Modular machine learning pipeline for cross-jurisdiction legal auditing.
# ==============================================================================

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score

# Import generate_case to generate perturbed test data
from dataset_generator import generate_case

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# List of 20 countries
COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Australia", "Singapore",
    "India", "Germany", "France", "Italy", "Spain", "Switzerland", "Brazil",
    "Norway", "Sweden", "Denmark", "Japan", "South Korea", "Saudi Arabia",
    "Iran", "South Africa"
]

# Curated real-world validation cases
REAL_WORLD_VALIDATION_CASES = [
    {
        "name": "Armed Hijacking",
        "text": "The defendant hijacked a commercial cargo delivery truck at gunpoint, binding the driver's hands and stealing high-value electronics worth $45,000 before fleeing."
    },
    {
        "name": "School Fund Embezzlement",
        "text": "A high school teacher embezzled $3,200 from the student activities club fund over a six-month period, submitting falsified receipts to hide the transactions."
    },
    {
        "name": "Petty Shoplifting",
        "text": "The suspect entered a convenience store, slipped three bottles of premium whiskey under his coat, and walked out without paying."
    },
    {
        "name": "Spontaneous Bar Assault",
        "text": "The defendant engaged in a sudden, alcohol-fueled bar fight, striking the victim in the face with a heavy beer bottle, causing a fractured jaw."
    },
    {
        "name": "Large-Scale Drug Smuggling",
        "text": "The offender was intercepted at the commercial port smuggling 12 kilograms of pure methamphetamines concealed inside hollowed-out wooden furniture."
    }
]

def load_data(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def extract_text_embeddings(texts, batch_size=32):
    print("Loading HuggingFace transformer model 'sentence-transformers/all-MiniLM-L6-v2'...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Extracting text embeddings on {device} (batch size: {batch_size})...")
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = model(**encoded)
            
        token_embeddings = output[0]
        attention_mask = encoded["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        embeddings.append(batch_embeddings)
        
    return np.vstack(embeddings)

def train_severity_model(embeddings, df):
    print("Training Model 1: Severity Mapping Model...")
    targets = ["violence_score", "financial_loss_score", "victim_vulnerability_score", "premeditation_score", "public_safety_risk"]
    Y = df[targets].values
    X = embeddings
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X, Y)
    
    preds = model.predict(X)
    r2 = r2_score(Y, preds, multioutput="uniform_average")
    print(f"Model 1 Severity Regressor trained. Average R^2 score: {r2:.4f}")
    
    return model

def validate_nlp_on_real_cases(severity_model):
    print("\n--- Running NLP Model 1 Validation on Real-World Case Summaries ---")
    texts = [case["text"] for case in REAL_WORLD_VALIDATION_CASES]
    embeddings = extract_text_embeddings(texts)
    predictions = severity_model.predict(embeddings)
    predictions = np.clip(predictions, 0.0, 1.0)
    
    for idx, case in enumerate(REAL_WORLD_VALIDATION_CASES):
        pred = predictions[idx]
        print(f"\nCase Name: {case['name']}")
        print(f"Text: '{case['text']}'")
        print(f"Extracted Severity: Violence={pred[0]:.2f}, Financial={pred[1]:.2f}, Vulnerability={pred[2]:.2f}, Premeditation={pred[3]:.2f}, Safety={pred[4]:.2f}")
    print("-------------------------------------------------------------------\n")

def train_sentencing_models(df, severity_preds):
    print("Training Model 2: Country-Specific Sentence Predictors...")
    # Features for GBDT: 5 severity predictions + 5 legal/case factors
    feature_cols = ["priors", "plea_guilty", "mitigating_circumstances", "juvenile", "court_region"]
    X = np.hstack([
        severity_preds, 
        df[feature_cols].values
    ])
    
    # 1. Generate formula-shifted held-out test data for leakage check (200 cases)
    print("Generating 200 formula-shifted perturbed cases for leakage check...")
    shifted_cases = [generate_case(2000 + i, perturb=True) for i in range(200)]
    shifted_df = pd.DataFrame(shifted_cases)
    
    # Extract embeddings for shifted cases
    shifted_texts = shifted_df["fact_pattern"].tolist()
    shifted_embeddings = extract_text_embeddings(shifted_texts)
    
    # Run severity model on shifted embeddings
    # (Since this is a validation step, we import the severity model we just trained)
    # We load the severity mapping directly
    global_severity_model = joblib.load(os.path.join("models", "severity_model.pkl")) if os.path.exists(os.path.join("models", "severity_model.pkl")) else None
    
    country_models = {}
    country_thresholds = {}
    
    # We will train the severity model first and save it, so we can load it here.
    # To avoid loading errors, let's pass the trained severity model directly to this function.
    
    # Let's project shifted cases into severity space
    shifted_severity_preds = train_severity_model_obj.predict(shifted_embeddings)
    shifted_severity_preds = np.clip(shifted_severity_preds, 0.0, 1.0)
    
    X_test = np.hstack([
        shifted_severity_preds,
        shifted_df[feature_cols].values
    ])
    
    print("\n--- Leakage and Quantile Anomaly Boundary Calculations ---")
    for country in COUNTRIES:
        y = df[f"sentence_{country}"].values
        
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate training performance (formula fitting)
        train_preds = model.predict(X)
        train_r2 = r2_score(y, train_preds)
        
        # Evaluate held-out perturbed test performance (generalizability under shift)
        y_test = shifted_df[f"sentence_{country}"].values
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        
        # Calculate absolute residuals on training set for empirical quantiles
        abs_residuals = np.abs(y - train_preds)
        q90 = float(np.percentile(abs_residuals, 90))
        q95 = float(np.percentile(abs_residuals, 95))
        
        country_models[country] = model
        country_thresholds[country] = {
            "q90": q90,
            "q95": q95
        }
        
        print(f"  - {country}:")
        print(f"    Train R^2 (Formula Fit): {train_r2:.4f} | Held-out Perturbed Test R^2 (Generalization): {test_r2:.4f}")
        print(f"    Empirical Residual Thresholds -> 90th Pct: {q90:.2f}m, 95th Pct: {q95:.2f}m")
        
    print("-----------------------------------------------------------\n")
    return country_models, country_thresholds

def build_clustering_and_pca(country_models):
    print("Training Model 3: Country Sentencing Typology Map...")
    
    # 100 benchmark cases: 5 severity + 5 legal factors
    grid_size = 100
    np.random.seed(123)
    
    benchmark_cases = []
    for _ in range(grid_size):
        violence = np.random.uniform(0.0, 1.0)
        financial = np.random.uniform(0.0, 1.0)
        vulnerability = np.random.uniform(0.0, 1.0)
        premeditation = np.random.uniform(0.0, 1.0)
        safety = np.random.uniform(0.0, 1.0)
        
        priors = np.random.choice([0, 1, 2, 3])
        plea = np.random.choice([0, 1])
        mitigation = np.random.choice([0, 1])
        juvenile = np.random.choice([0, 1], p=[0.9, 0.1])
        region = np.random.choice([0, 1]) # Rural/Metro
        
        benchmark_cases.append([violence, financial, vulnerability, premeditation, safety, priors, plea, mitigation, juvenile, region])
        
    benchmark_cases = np.array(benchmark_cases)
    
    # Get country sentencing profiles (predictions on benchmark grid)
    country_profiles = []
    for country in COUNTRIES:
        model = country_models[country]
        preds = model.predict(benchmark_cases)
        country_profiles.append(preds)
        
    country_profiles = np.array(country_profiles) # (20, 100)
    
    # Run dynamic Silhouette Analysis to find optimal K (between 2 and 7)
    print("\n--- Running Silhouette Analysis for Optimal Clusters (K) ---")
    best_k = 3
    best_score = -1
    silhouette_scores = {}
    
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(country_profiles)
        score = silhouette_score(country_profiles, labels)
        silhouette_scores[k] = score
        print(f"  K = {k} | Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal clusters selected: K = {best_k} (highest silhouette score: {best_score:.4f})")
    print("------------------------------------------------------------\n")
    
    # Perform final K-Means with optimal K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(country_profiles)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(country_profiles)
    
    # Dynamically label and describe clusters based on average sentence lengths
    cluster_averages = {}
    for i in range(best_k):
        cluster_averages[i] = np.mean(country_profiles[cluster_labels == i])
        
    sorted_clusters = sorted(cluster_averages.items(), key=lambda x: x[1])
    
    # Dynamic Cluster Mapping
    cluster_mapping = {}
    for rank, (cluster_idx, avg_val) in enumerate(sorted_clusters):
        if rank == 0:
            cluster_mapping[cluster_idx] = {
                "id": 0,
                "name": "Rehabilitative (Low-Severity / Flat)",
                "description": "System characterized by short, flat sentencing curves, low overall prison terms, and a heavy focus on rehabilitation."
            }
        elif rank == len(sorted_clusters) - 1:
            cluster_mapping[cluster_idx] = {
                "id": len(sorted_clusters) - 1,
                "name": "Deterrence-Steep (High-Severity / Punitive)",
                "description": "Punitive system with steep escalation curves, high baseline penalties, heavy priors multipliers, and limited mitigation flexibility."
            }
        else:
            cluster_mapping[cluster_idx] = {
                "id": rank,
                "name": f"Proportional Level {rank} (Balanced Civil/Common Law)",
                "description": "System maintaining moderate penalties scaled proportionally to crime severity, with standardized legal discounts for plea and mitigation."
            }
            
    # Finalize country typology metadata
    typology_data = {}
    for idx, country in enumerate(COUNTRIES):
        raw_label = cluster_labels[idx]
        mapped = cluster_mapping[raw_label]
        
        typology_data[country] = {
            "x": float(pca_coords[idx, 0]),
            "y": float(pca_coords[idx, 1]),
            "cluster_id": mapped["id"],
            "cluster_name": mapped["name"],
            "cluster_description": mapped["description"]
        }
        print(f"  - {country} clustered into '{mapped['name']}' at coords ({pca_coords[idx, 0]:.2f}, {pca_coords[idx, 1]:.2f})")
        
    return typology_data, benchmark_cases.tolist()

# Global variable to pass the trained model to the GBDT training function
train_severity_model_obj = None

def main():
    global train_severity_model_obj
    os.makedirs("models", exist_ok=True)
    
    csv_path = os.path.join("ml", "synthetic_cases.csv")
    df = load_data(csv_path)
    
    # 1. Extract text embeddings
    texts = df["fact_pattern"].tolist()
    embeddings = extract_text_embeddings(texts)
    
    # 2. Train Severity Embedding Model (Model 1)
    train_severity_model_obj = train_severity_model(embeddings, df)
    
    # Save Model 1 first, so we have it stored
    joblib.dump(train_severity_model_obj, os.path.join("models", "severity_model.pkl"))
    
    # 3. Validate severity model on real case texts
    validate_nlp_on_real_cases(train_severity_model_obj)
    
    # Predict severity scores on training set to feed into sentencing model
    severity_preds = train_severity_model_obj.predict(embeddings)
    severity_preds = np.clip(severity_preds, 0.0, 1.0)
    
    # 4. Train Country-Specific Sentencing Predictors (Model 2) with Perturbed held-out set
    country_models, country_thresholds = train_sentencing_models(df, severity_preds)
    
    # 5. Perform functional clustering and PCA (Model 3) with dynamic K selection
    typology_data, benchmark_grid = build_clustering_and_pca(country_models)
    
    # 6. Save remaining models and metadata
    print("Saving trained models and metadata...")
    joblib.dump(country_models, os.path.join("models", "country_models.pkl"))
    joblib.dump(country_thresholds, os.path.join("models", "country_residuals_std.pkl"))
    
    typology_metadata = {
        "typology": typology_data,
        "benchmark_grid": benchmark_grid
    }
    joblib.dump(typology_metadata, os.path.join("models", "typology_metadata.pkl"))
    
    print("End-to-end training pipeline successfully completed and all models saved.")

if __name__ == "__main__":
    main()
