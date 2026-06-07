## 🧠 Backend Architecture: The Aegis AI Engine

The Aegis AI backend is not a standard rule-based filter; it is a **Two-Brain Machine Learning Pipeline** built for real-time transaction interception. It is served via a lightning-fast asynchronous REST API designed to process payloads in **<40ms**.

### 🛠️ Core Tech Stack
* **Framework:** FastAPI (Python)
* **Supervised Learning:** XGBoost (`xgboost`)
* **Unsupervised Learning:** Isolation Forest (`scikit-learn`)
* **Data Manipulation:** Pandas & NumPy
* **Model Serialization:** Joblib

### ⚙️ The "Two-Brain" ML Pipeline
Financial criminals continuously evolve to bypass rigid banking thresholds (e.g., micro-structuring and smurfing). To combat zero-day threats while recognizing historical fraud patterns, Aegis AI utilizes a dual-layer approach:

1. **Layer 1: The Detective (Isolation Forest)**
   * **Type:** Unsupervised Anomaly Detection.
   * **Purpose:** Analyzes the baseline transaction behavior and flags statistical impossibilities or bizarre transactional velocities without requiring prior training labels. Outputs a binary anomaly state (0.0 or 1.0).

2. **Layer 2: The Enforcer (XGBoost)**
   * **Type:** Supervised Gradient Boosting.
   * **Purpose:** Trained on thousands of historical banking records to hunt down specific behavioral signatures, such as high "Pass-Through Ratios" indicative of Money Mules. Outputs a precise fraud probability percentage.

3. **Baseline Imputation**
   * Engineered to prevent the "Missing Value Trap" that breaks standard synthetic data bots. Missing categorical features in the payload are instantly mapped to their historical baseline medians before inference.

### 🧮 The Ensemble Equation
The final threat assessment is fused using a custom weighted ensemble equation. We cap the supervised model's authority to prevent false positives on specific single-feature triggers, requiring the unsupervised layer to co-sign on extreme anomalies.

$R_{score} = 0.7 \cdot P_{supervised} + 0.3 \cdot S_{anomaly}$

*Where $R_{score}$ is the Final Risk Percentage, $P_{supervised}$ is the XGBoost probability, and $S_{anomaly}$ is the Isolation Forest trigger.*

### 🚀 API Reference

**Endpoint:** `POST /analyze`

**Description:** Ingests a JSON payload containing transaction features, runs inference through both models, applies the ensemble equation, and returns a dynamic action directive (`APPROVE`, `REVIEW`, or `FREEZE`).

**Sample Request Payload:**
```json
{
  "features": {
    "F3912": 1,
    "F2230": 99999,
    "F1922": -88888,
    "F3799": 9999999
  }
}