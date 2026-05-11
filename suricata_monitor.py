import json
import joblib
import pandas as pd
from datetime import datetime

# ==============================
# LOAD MODEL
# ==============================

print("Loading XGBoost model...")

model = joblib.load(
    "model/xgboost_model.pkl"
)

# ==============================
# LOAD SCALER
# ==============================

print("Loading scaler...")

scaler = joblib.load(
    "model/xgb_scaler.pkl"
)

# ==============================
# LOAD FEATURE NAMES
# ==============================

print("Loading feature names...")

feature_names = joblib.load(
    "model/feature_names.pkl"
)

print("🚀 Suricata AI Monitor Started Successfully!")

# ==============================
# LOG FILE
# ==============================

LOG_FILE = "/var/log/suricata/eve.json"

print(f"📂 Monitoring log file: {LOG_FILE}")

# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(event):

    try:

        # Create empty feature vector
        features = [0] * len(feature_names)

        # Map important Suricata fields
        feature_map = {

            "Flow Duration":
                event.get("flow", {}).get(
                    "age", 0
                ),

            "Total Fwd Packets":
                event.get("flow", {}).get(
                    "pkts_toserver", 0
                ),

            "Total Backward Packets":
                event.get("flow", {}).get(
                    "pkts_toclient", 0
                ),

            "Total Length of Fwd Packets":
                event.get("flow", {}).get(
                    "bytes_toserver", 0
                ),

            "Total Length of Bwd Packets":
                event.get("flow", {}).get(
                    "bytes_toclient", 0
                ),

            "Protocol":
                6 if event.get("proto") == "TCP"
                else 17 if event.get("proto") == "UDP"
                else 1 if event.get("proto") == "ICMP"
                else 0,
        }

        # Fill matching features
        for i, feature_name in enumerate(feature_names):

            if feature_name in feature_map:

                features[i] = feature_map[
                    feature_name
                ]

        return features

    except Exception as e:

        print("❌ Feature Extraction Error:", e)

        return None

# ==============================
# MONITOR FUNCTION
# ==============================

def monitor():

    print("🎯 Waiting for Suricata events...")

    with open(LOG_FILE, "r") as f:

        # Move to end of file
        f.seek(0, 2)

        while True:

            line = f.readline()

            # No new line
            if not line:
                continue

            line = line.strip()

            # Empty line
            if not line:
                continue

            try:

                # Safe JSON parsing
                try:

                    event = json.loads(line)

                except json.JSONDecodeError:

                    continue

                # Only process flow events
                if event.get("event_type") == "flow":

                    print("📦 Suricata Event Captured")

                    features = extract_features(
                        event
                    )

                    if features is not None:

                        # Convert to DataFrame
                        features_df = pd.DataFrame(
                            [features],
                            columns=feature_names
                        )

                        # Scale features
                        scaled_features = scaler.transform(
                            features_df
                        )

                        # Predict
                        prediction = model.predict(
                            scaled_features
                        )[0]

                        timestamp = datetime.now()

                        # ==============================
                        # ATTACK DETECTED
                        # ==============================

                        if prediction == 1:

                            print(
                                f"🚨 ATTACK DETECTED : {timestamp}"
                            )

                            with open(
                                "logs/alerts.txt",
                                "a"
                            ) as alert_file:

                                alert_file.write(
                                    f"🚨 ATTACK DETECTED : {timestamp}\n"
                                )

                        # ==============================
                        # NORMAL TRAFFIC
                        # ==============================

                        else:

                            print(
                                f"✅ Normal Traffic : {timestamp}"
                            )

            except Exception as e:

                print("❌ Error:", e)

# ==============================
# START MONITORING
# ==============================

monitor()