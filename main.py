from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

print("Booting up Aegis AI Core Server...")
app = FastAPI(title="Aegis AI Core Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    aegis_engine = joblib.load('aegis_engine.pkl')
    xgb_model = aegis_engine['xgboost']
    iso_forest = aegis_engine['isolation_forest']
    feature_names = aegis_engine['features']
    baseline_profile = aegis_engine['baseline'] # LOAD THE NORMAL BASELINE
    print("✅ Flawless Aegis Engine Loaded!")
except Exception as e:
    print(f"❌ FATAL ERROR loading engine: {e}")
    
class TransactionData(BaseModel):
    features: dict
    
@app.post("/analyze")    
def analyze_transaction(data: TransactionData):
    # 1. Start with a perfect, safe, average human transaction
    transaction_dict = baseline_profile.copy()
    
    # 2. Overwrite it with ONLY the features sent from the frontend
    transaction_dict.update(data.features)
    
    # 3. Convert to DataFrame
    input_df = pd.DataFrame([transaction_dict], columns=feature_names)
    
    # --- The Aegis Tri-Layer Math Formula ---
    weight_supervised = 0.7
    
    p_supervised = xgb_model.predict_proba(input_df)[:, 1][0]
    
    raw_anomaly = iso_forest.predict(input_df)[0]
    s_anomaly = 1.0 if raw_anomaly == -1 else 0.0
    
    r_score = (weight_supervised * p_supervised) + ((1 - weight_supervised) * s_anomaly)
    final_risk_percentage = round(r_score * 100, 2)
    
    if final_risk_percentage > 80:
        action = "FREEZE"
    elif final_risk_percentage > 50:
        action = "REVIEW"
    else:
        action = "APPROVE"
        
    # Force all NumPy types to native Python types before sending to React
    return {
        "risk_score": round(float(final_risk_percentage), 2),
        "action": str(action),
        "layer_2_supervised": round(float(xgboost_prob * 100), 2),
        "layer_1_anomaly": float(anomaly_pred[0])  # Or however your anomaly variable is named
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
