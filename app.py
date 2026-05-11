from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ============================================
# LOAD MODEL
# ============================================

print("Loading XGBoost model...")

model = joblib.load(
    "model/xgboost_model.pkl"
)

# ============================================
# LOAD SCALER
# ============================================

print("Loading scaler...")

scaler = joblib.load(
    "model/xgb_scaler.pkl"
)

# ============================================
# LOAD FEATURE NAMES
# ============================================

print("Loading feature names...")

feature_names = joblib.load(
    "model/feature_names.pkl"
)

# ============================================
# CREATE FLASK APP
# ============================================

app = Flask(__name__)

# ============================================
# HOME ROUTE
# ============================================

@app.route("/")

def home():

    return jsonify({

        "message":
        "🚀 AI-Based Hybrid IDPS Running Successfully!"

    })

# ============================================
# PREDICTION ROUTE
# ============================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # ====================================
        # RECEIVE JSON DATA
        # ====================================

        data = request.json

        # ====================================
        # GET FEATURES
        # ====================================

        features = data["features"]

        # ====================================
        # VALIDATE FEATURE LENGTH
        # ====================================

        if len(features) != len(feature_names):

            return jsonify({

                "error":
                f"Expected {len(feature_names)} features"

            })

        # ====================================
        # CREATE DATAFRAME
        # ====================================

        features_df = pd.DataFrame(

            [features],

            columns=feature_names
        )

        # ====================================
        # SCALE FEATURES
        # ====================================

        scaled_features = scaler.transform(
            features_df
        )

        # ====================================
        # PREDICT
        # ====================================

        prediction = model.predict(
            scaled_features
        )[0]

        # ====================================
        # CONVERT RESULT
        # ====================================

        result = (

            "ATTACK"

            if prediction == 1

            else "NORMAL"

        )

        # ====================================
        # RETURN RESPONSE
        # ====================================

        return jsonify({

            "prediction": result

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# ============================================
# START SERVER
# ============================================

if __name__ == "__main__":

    print("\n🚀 Flask API Started Successfully!")

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )