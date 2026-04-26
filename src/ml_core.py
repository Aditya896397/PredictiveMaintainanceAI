import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def generate_training_data(self):
        # Synthetic data generation for initial training
        np.random.seed(42)
        n_samples = 1500
        
        # Normal ops
        temp_normal = np.random.normal(50, 10, int(n_samples*0.8))
        vib_normal = np.random.normal(0.3, 0.1, int(n_samples*0.8))
        pressure_normal = np.random.normal(100, 15, int(n_samples*0.8))
        target_normal = np.zeros(int(n_samples*0.8))
        
        # Anomalous ops
        temp_anom = np.random.normal(90, 15, int(n_samples*0.2))
        vib_anom = np.random.normal(1.5, 0.4, int(n_samples*0.2))
        pressure_anom = np.random.normal(140, 20, int(n_samples*0.2))
        target_anom = np.ones(int(n_samples*0.2))
        
        X = pd.DataFrame({
            'temperature': np.concatenate([temp_normal, temp_anom]),
            'vibration': np.concatenate([vib_normal, vib_anom]),
            'pressure': np.concatenate([pressure_normal, pressure_anom]),
        })
        y = np.concatenate([target_normal, target_anom])
        return X, y

    def train(self):
        X, y = self.generate_training_data()
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict_risk(self, telemetry_df):
        if not self.is_trained:
            self.train()
            
        features = telemetry_df[['temperature', 'vibration', 'pressure']]
        probas = self.model.predict_proba(features)[:, 1] # Probability of class 1 (failure)
        
        results = []
        for idx, row in telemetry_df.iterrows():
            prob = probas[idx]
            risk_level = "Low"
            if prob > 0.7:
                risk_level = "Critical"
            elif prob > 0.35:
                risk_level = "Elevated"
                
            results.append({
                "machine_id": row['machine_id'],
                "failure_probability": prob,
                "risk_level": risk_level
            })
            
        return pd.DataFrame(results)

    def generate_diagnostic(self, state, risk_prob):
        if risk_prob < 0.35:
            return "Operating within optimum parameters. No action required."
            
        diagnosis = []
        if state['temperature'] > 75:
            diagnosis.append("Elevated thermal signature detected.")
        if state['vibration'] > 0.8:
            diagnosis.append("Acute mechanical vibration spike observed.")
        if state['pressure'] > 120:
            diagnosis.append("Abnormal pressure levels.")
            
        # Agentic framing
        action = "Agent Recommendation: Schedule standard preventative inspection within 7 days."
        if risk_prob > 0.7:
            if state['vibration'] > 0.8:
                action = "EMERGENCY: Halt operation to prevent catastrophic bearing/motor destruction. Immediate inspection required. Est downtime saved: 8 hrs."
            elif state['temperature'] > 75:
                action = "CRITICAL: Potential cooling system failure or severe friction. Assess thermal load immediately. Est repair cost saved: $12k."
                
        return "[Agent Analysis] " + " ".join(diagnosis) + " \n\n" + action
