# Global Sentencing Proportionality Drift Auditor

An advanced machine learning-driven compliance and auditing platform built to model, compare, and audit sentencing patterns across 20 representative jurisdictions. Using continuous harm embedding, multi-variable regression, and silhouette-optimized clustering, this auditor measures relative deviation (Proportionality Drift) and within-jurisdiction anomalies without enforcing moral baselines or prescribing "correct" sentences.

**Author:** Mohibul Hoque (<hokworks@gmail.com>) | [GitHub](https://github.com/speedyhok) | [LinkedIn](https://linkedin.com/in/speedymohibul)  
**System:** Flask / Sentence-Transformers / GBDT Pipeline  
**Link of the live website:** https://sentencing-punishment-predictor-in-sp47.onrender.com

---

## 🛠️ Multi-Tier Machine Learning Architecture

The system utilizes a **three-tier sequential machine learning pipeline** to process crime narratives and generate jurisdictional predictions.

```
+---------------------------+
|  Crime Fact Pattern Text  |
+-------------+-------------+
              |
              v
+-------------+-------------+
| Model 1: NLP Transformer  |  <--- (Sentence-Transformers all-MiniLM-L6-v2)
+-------------+-------------+
              | (384D Embedding Vector)
              v
+-------------+-------------+
|  MLP Regressor Severity   |  <--- (Multi-Layer Perceptron Severity Mapper)
+-------------+-------------+
              | (5D Continuous Severity Vector:
              |  Violence, Property Loss, Vulnerability, Premeditation, Safety Risk)
              +-----------------------+
              |                       |
              v                       v
+-------------+-------------+  +------+------+
| Model 2: Country GBDTs    |  | Case Toggles| <--- (Priors, Plea, Mitigating,
+-------------+-------------+  +------+------+       Juvenile, Metropolitan)
              |
              +-----------------------+
              | (20 Predicted Sentences)
              v
+-------------+-------------+
|  Reference Median (20)    |  <--- (Anomaly Detection Gauges & Cross-Drift)
+-------------+-------------+
              |
              v
+-------------+-------------+
| Model 3: PCA & K-Means    |  <--- (Sentencing Typology Clustering Plot)
+---------------------------+
```

### 🧠 Model Tier 1: NLP Severity Mapping (Semantic Text Embedding + MLP)
* **What it does:** Extracts 5 continuous severity scores (scale `0.0` to `1.0`) directly from free-text crime narratives, independent of statutory codes:
  1. **Violence:** Physical harm, force, weapon usage, and bodily injury.
  2. **Property Loss:** Financial damage, embezzlement value, or volume of goods.
  3. **Victim Vulnerability:** Age (elderly, youth), health, dependency, or trust exploitation.
  4. **Premeditation:** Planning, stalking, coordination, or long-term scheming.
  5. **Public Safety Risk:** Spread of danger, illegal contraband volume, or community threat.
* **Why we chose this model:**
  * **Sentence Transformers (`all-MiniLM-L6-v2`):** Converts raw text into a dense 384-dimensional semantic vector. This model is extremely lightweight, fast, performs exceptionally well at semantic similarity, and can run in real-time on standard CPU resources.
  * **Multi-Layer Perceptron (MLP) Regressor:** Maps the 384D text embedding to the 5D continuous severity space. An MLP is chosen over linear regression because the relationship between semantic text patterns and multi-dimensional crime severity is highly non-linear. The trained MLP serves as a normalized "Severity Mapper" mapping diverse vocabulary patterns to a unified scale.

### 🌲 Model Tier 2: Country-Specific Sentence Prediction (Gradient Boosted Decision Trees - GBDT)
* **What it does:** Predicts expected sentence lengths (in months) for a given crime based on the 5 continuous severity scores plus 5 legal factors:
  1. **Prior Offenses:** Number of prior criminal records (`0` to `5`).
  2. **Early Guilty Plea / Confession:** Binary toggle.
  3. **Mitigating Circumstances:** Presence of mitigating factors (cooperation, duress).
  4. **Offender is Juvenile:** Minor status binary toggle.
  5. **Metropolitan Court Location:** Regional bias binary toggle (Metropolitan vs. Rural).
* **Why we chose this model:**
  * **Gradient Boosted Decision Trees (GBDTs via `HistGradientBoostingRegressor`):** We train 20 independent country GBDT models. GBDTs are selected because:
    - **Tabular Superiority:** GBDTs are the gold standard for heterogeneous tabular datasets (mixing continuous severity scores with binary/integer legal factors).
    - **Non-Linear Thresholds:** Legal codes and guidelines feature sharp, non-linear thresholds (e.g., priors cap out, mandatory minimums trigger under specific circumstances, plea discount caps). GBDTs handle these step-functions and non-linear interactions natively, whereas linear models struggle.
    - **Robustness to Overfitting:** Combined with regularization, GBDTs generalize well to perturbed test sets (yielding high $R^2$ on held-out test data) instead of memorizing equations.

### 🗺️ Model Tier 3: Jurisdictional Typology Clustering (PCA + K-Means)
* **What it does:** Discovers latent structural groupings of legal systems based on their functional behavior (how they scale expected sentences in response to crime severity).
* **Why we chose this model:**
  * **Principal Component Analysis (PCA):** Reduces the dimensionality of the 20-country prediction profiles to 2 components (Slope/Severity Scaling vs. Mitigation Variance) for visualization on a 2D scatter plot.
  * **K-Means with Dynamic Silhouette Score Selection:** Automatically groups countries into typologies. Rather than hardcoding the number of clusters ($K$), K-Means runs inside a dynamic loop evaluating silhouette scores for $K \in [2..7]$, selecting the $K$ that maximizes cohesion and separation. This prevents arbitrary clustering, resulting in an optimal $K=2$ (Deterrence-Steep vs. Rehabilitative).

---

## 📐 Mathematical Formulations

To ensure auditing rigor and prevent statistical distortions, the system implements the following mathematical improvements:

### 1. Multiplicative Log-Normal Noise
Instead of simple additive Gaussian noise (which distorts small sentences and can yield negative sentence lengths), synthetic data is generated with multiplicative log-normal noise:
$$y_{\text{sim}} = y_{\text{base}} \times e^{\epsilon}, \quad \epsilon \sim N(0, \sigma_{\text{noise}})$$
This ensures that variance scales proportionally with sentence length (minor crimes get low absolute variance, severe crimes get wide absolute variance) and guarantees that all simulated sentences remain non-negative.

### 2. Empirical Quantile Anomaly Boundaries (Within-Jurisdiction Drift)
To score within-jurisdiction consistency, we do not assume residuals are normally distributed. We calculate the distribution of absolute residuals on the training set:
$$\text{Residual}_i = |y_i - \hat{y}_i|$$
We compute the **90th percentile** (Moderate Drift threshold) and **95th percentile** (Critical Drift threshold) for each country independently. When auditing an actual sentence, we compare the absolute residual against these empirical boundaries.

### 3. Zero-Guarded Cross-Jurisdiction Drift Index
To compute how much a specific country's predicted sentence deviates from the global baseline (Reference Median) without creating mathematical distortions for short sentences, we use a zero-guarded percentage deviation formula:
$$\text{Drift}_{\text{cross}} = \frac{\hat{y}_{\text{country}} - \text{Median}(\mathbf{\hat{y}})}{\max(\text{Median}(\mathbf{\hat{y}}), 10^{-6})} \times 100\%$$

---

## 💻 Web Auditor Dashboard Features

The web frontend includes a responsive, glassmorphic UI built with Vanilla CSS and JS:

1. **Case Parameters Panel:** Configures the crime narrative, priors count, guilty plea, mitigation, juvenile status, and court region (Metropolitan vs. Rural).
2. **Readability Optimization:** Fully dark-mode compatible with elevated card backgrounds (`rgba(13, 18, 30, 0.96)`) and high-contrast light slate text variables to ensure maximum legibility.
3. **ML Pipeline Inference Trace:** A real-time trace card that displays:
   - The first 8 dimensions of the online 384D transformer text embedding.
   - The labeled 10-feature GBDT input vector.
   - The audited country's 2D PCA projection coordinates and cluster details.
4. **Sentencing Typology Map:** An interactive 2D scatter plot where clicking on country dots automatically synchronizes the audit selectors, gauges, and ML trace coordinates in real-time.

---

## ⚙️ How to Run the Project

### Prerequisites
Install Python 3.8+ and the required packages:
```bash
pip install flask numpy torch transformers scikit-learn joblib reportlab
```

### 1. Train the Models
Generate the dataset, check for data leakage, optimize $K$, and train the MLP and country GBDTs:
```bash
python ml/train_pipeline.py
```
This generates the model files and saves them to the `models/` directory.

### 2. Launch the Web Application
Run the Flask server:
```bash
python app.py
```

### 3. Open the Dashboard
Open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Author & Copyright
Developed and designed by **Mohibul Hoque** (<hokworks@gmail.com>) - [GitHub](https://github.com/speedyhok) | [LinkedIn](https://linkedin.com/in/speedymohibul) (2026). 
This project is part of an ongoing architecture framework for algorithmic legal auditing.
