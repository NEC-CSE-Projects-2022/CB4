# # backend/app.py
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import os
# import traceback
# import pickle
# # model loading utilities
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model # type: ignore
# import xgboost as xgb
# import lightgbm as lgb



# app = Flask(__name__)
# CORS(app)  # allow cross-origin requests (adjust in production)

# # ---- Configuration: set model paths (update if different) ----
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# # Example filenames - put your real model files here or mount Drive in Colab
# GRU_MODEL_PATH = os.path.join(MODEL_DIR, "models/gru_model.h5")
# TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "models/transformer_model.h5")
# XGB_MODEL_PATH = os.path.join(MODEL_DIR, "models/xgb_model.pkl")   # or .bin / .pkl
# LGB_MODEL_PATH = os.path.join(MODEL_DIR, "models/lgb_model.pkl")    # or .pkl
# META_MODEL_PATH = os.path.join(MODEL_DIR, "models/meta_cat_model.pkl")  # meta learner (CatBoost/MLP saved)

# # ---- Load models (safe: try/except so server still runs if missing) ----
# print("Loading models...")
# gru_model = None
# transformer_model = None
# xgb_model = None
# lgb_model = None
# meta_model = None

# try:
#     if os.path.exists(GRU_MODEL_PATH):
#         gru_model = load_model(GRU_MODEL_PATH)
#         print("Loaded GRU model")
# except Exception as e:
#     print("GRU load error:", e)

# try:
#     if os.path.exists(TRANSFORMER_MODEL_PATH):
#         transformer_model = load_model(TRANSFORMER_MODEL_PATH)
#         print("Loaded Transformer model")
# except Exception as e:
#     print("Transformer load error:", e)

# try:
#     if os.path.exists(XGB_MODEL_PATH):
#         # xgb can load from json or binary - we attempt both
#         xgb_model = xgb.XGBRegressor()
#         xgb_model.load_model(XGB_MODEL_PATH)
#         print("Loaded XGBoost model")
# except Exception as e:
#     print("XGB load error:", e)

# try:
#     if os.path.exists(LGB_MODEL_PATH):
#         # LightGBM python API load if saved as txt or pickle
#         lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
#         print("Loaded LightGBM model")
# except Exception as e:
#     print("LightGBM load error:", e)

# try:
#     if os.path.exists(META_MODEL_PATH):
#         meta_model = joblib.load(META_MODEL_PATH)
#         print("Loaded Meta model")
# except Exception as e:
#     print("Meta model load error:", e)

# # ---- Helper: format features ----
# def prepare_sequence_input(x_array):
#     # x_array shape -> (n_features,) for single sample
#     # Many of your sequence models expect shape (batch, timesteps, features)
#     # This simple wrapper will reshape to (1, 1, n_features) for 1-step sequence.
#     arr = np.array(x_array, dtype=np.float32)
#     arr = arr.reshape(1, 1, -1)  # adjust if your model expects longer windows
#     return arr

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"status": "ok", "loaded": {
#         "gru": bool(gru_model),
#         "transformer": bool(transformer_model),
#         "xgb": bool(xgb_model),
#         "lgb": bool(lgb_model),
#         "meta": bool(meta_model),
#     }})

# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Expect JSON:
#     {
#       "features": [v1, v2, v3, ...],   # exactly same order used during training
#       "feature_names": ["pH","DO","BOD","TEMP", ...]   # optional
#     }
#     Returns:
#     {
#       "predictions": {"gru": x, "xgb": y, "lgb": z, "transformer": t, "meta": final}
#     }
#     """
#     try:
#         data = request.get_json(force=True)
#         features = data.get("features", None)
#         if features is None:
#             return jsonify({"error": "No 'features' provided"}), 400

#         # Convert to numpy array
#         x = np.array(features, dtype=float).reshape(1, -1)

#         preds = {}
#         # GRU
#         if gru_model is not None:
#             x_seq = prepare_sequence_input(features)
#             p_gru = float(gru_model.predict(x_seq, verbose=0).flatten()[0])
#             preds["gru"] = p_gru

#         # Transformer
#         if transformer_model is not None:
#             x_seq = prepare_sequence_input(features)
#             p_trans = float(transformer_model.predict(x_seq, verbose=0).flatten()[0])
#             preds["transformer"] = p_trans

#         # XGBoost
#         if xgb_model is not None:
#             p_xgb = float(xgb_model.predict(x))
#             preds["xgb"] = p_xgb

#         # LightGBM (if loaded as Booster)
#         if lgb_model is not None:
#             # lgb.Booster.predict expects 2D array
#             p_lgb = float(lgb_model.predict(x))
#             preds["lgb"] = p_lgb

#         # Stack-level: meta model expects stacked preds or OOF-style features
#         if meta_model is not None:
#             # create stack vector in same ordering used to train meta-model
#             stack_vector = []
#             # keep consistent ordering
#             for key in ["gru", "transformer", "xgb", "lgb"]:
#                 stack_vector.append(preds.get(key, 0.0))
#             stack_arr = np.array(stack_vector).reshape(1, -1)
#             # If meta_model is sklearn-like
#             try:
#                 final = meta_model.predict(stack_arr)
#                 preds["meta"] = float(final.flatten()[0])
#             except Exception:
#                 # if CatBoost or other custom
#                 try:
#                     preds["meta"] = float(meta_model.predict(stack_arr))
#                 except Exception as ex:
#                     preds["meta_error"] = str(ex)

#         return jsonify({"predictions": preds})

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# if __name__ == "__main__":
#     # For Colab + ngrok: set host to 0.0.0.0 and port to 5000
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)








# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# import xgboost as xgb
# import lightgbm as lgb
# import joblib

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Allow frontend requests (React)

# # -----------------------
# # Load Pre-trained Models
# # -----------------------
# print("Loading models...")

# try:
#     gru_model = tf.keras.models.load_model("models/gru_model.h5", compile=False)
#     print("GRU model loaded successfully.")
# except Exception as e:
#     print("GRU load error:", e)
#     gru_model = None

# try:
#     transformer_model = tf.keras.models.load_model("models/transformer_model.h5", compile=False)
#     print("Transformer model loaded successfully.")
# except Exception as e:
#     print("Transformer load error:", e)
#     transformer_model = None

# try:
#     xgb_model = joblib.load("models/xgb_model.pkl")
#     print("XGBoost model loaded successfully.")
# except Exception as e:
#     print("XGB load error:", e)
#     xgb_model = None

# try:
#     lgb_model = joblib.load("models/lgb_model.pkl")
#     print("LightGBM model loaded successfully.")
# except Exception as e:
#     print("LightGBM load error:", e)
#     lgb_model = None

# try:
#     meta_model = joblib.load("models/meta_catboost.pkl")
#     print("Meta-CatBoost model loaded successfully.")
# except Exception as e:
#     print("Meta model load error:", e)
#     meta_model = None


# # ------------------------
# # Utility: Quality Category
# # ------------------------
# def quality_category(wqi):
#     if wqi >= 90:
#         return "Excellent 💧"
#     elif wqi >= 70:
#         return "Good 🌿"
#     elif wqi >= 50:
#         return "Moderate ⚠️"
#     else:
#         return "Poor ❌"


# # -----------------
# # Prediction Route
# # -----------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         features = np.array([
#             float(data["pH"]),
#             float(data["DO"]),
#             float(data["Temp"]),
#             float(data["BOD"]),
#             float(data["FS"]),
#             float(data["TC"]),
#             float(data["FC"]),
#             float(data["Cond"]),
#             float(data["NO3"])
#         ]).reshape(1, -1)

#         preds = []

#         # Deep models
#         if gru_model is not None:
#             gru_pred = float(gru_model.predict(features, verbose=0)[0][0])
#             preds.append(gru_pred)
#         if transformer_model is not None:
#             transformer_pred = float(transformer_model.predict(features, verbose=0)[0][0])
#             preds.append(transformer_pred)

#         # Tree-based models
#         if xgb_model is not None:
#             xgb_pred = float(xgb_model.predict(xgb.DMatrix(features))[0])
#             preds.append(xgb_pred)
#         if lgb_model is not None:
#             lgb_pred = float(lgb_model.predict(features)[0])
#             preds.append(lgb_pred)

#         # Meta Ensemble Prediction
#         if meta_model is not None and len(preds) >= 2:
#             final_pred = meta_model.predict(np.array(preds).reshape(1, -1))[0]
#         else:
#             final_pred = np.mean(preds) if preds else 0

#         # Generate response
#         return jsonify({
#             "Predicted_WQI": round(float(final_pred), 2),
#             "Category": quality_category(final_pred)
#         })

#     except Exception as e:
#         print("Prediction error:", e)
#         return jsonify({"error": str(e)}), 400


# @app.route("/")
# def home():
#     return jsonify({"message": "AquaNet-X Water Quality API Running"})


# if __name__ == "__main__":
#     print("✅ AquaNet-X Backend Running on http://127.0.0.1:5000")
#     app.run(host="0.0.0.0", port=5000, debug=True)






# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# def classify_wqi(wqi):
#     if wqi >= 90:
#         return "Excellent"
#     elif wqi >= 70:
#         return "Good"
#     elif wqi >= 50:
#         return "Moderate"
#     else:
#         return "Poor"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     # Extract features (same order as model expects)
#     features = [
#         data['pH'],
#         data['DO'],
#         data['Temp'],
#         data['BOD'],
#         data['FS'],
#         data['TC'],
#         data['FC'],
#         data['Cond'],
#         data['NO3']
#     ]

#     # Example prediction (replace with real model call)
#     gru_pred = gru_model.predict([features])
#     transformer_pred = transformer_model.predict([features])
#     xgb_pred = xgb_model.predict(np.array([features]))
#     lgb_pred = lgb_model.predict(np.array([features]))
#     meta_input = np.mean([gru_pred, transformer_pred, xgb_pred, lgb_pred])
#     final_pred = meta_catboost.predict(np.array([[meta_input]]))

    
#     # predicted_wqi = 89.2
#     # category = "Good" if predicted_wqi > 80 else "Poor"

#     return jsonify({
#         "Predicted_WQI": round(float(final_pred[0]), 2),
#         "Category": classify_wqi(final_pred[0])
#     })



# @app.route('/')
# def home():
#     return jsonify({"message": "AquaNet-X Water Quality API Running"})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)
CORS(app)  # Enables access from React frontend

# -------------------------------
# Load Models
# -------------------------------
print("Loading models...")

try:
    gru_model = load_model("models/gru_model.keras")
    print("GRU model loaded successfully.")
except Exception as e:
    print("Error loading GRU model:", e)
    gru_model = None

try:
    transformer_model = load_model("models/transformer_model.keras")
    print("Transformer model loaded successfully.")
except Exception as e:
    print("Error loading Transformer model:", e)
    transformer_model = None

try:
    xgb_model = joblib.load("models/xgb_model.pkl")
    print("XGBoost model loaded successfully.")
except Exception as e:
    print("Error loading XGBoost model:", e)
    xgb_model = None

try:
    lgb_model = joblib.load("models/lgb_model.pkl")
    print("LightGBM model loaded successfully.")
except Exception as e:
    print("Error loading LightGBM model:", e)
    lgb_model = None

try:
    scaler = joblib.load("models/scaler.pkl")
    print("Scaler model loaded successfully.")
except Exception as e:
    print("Error loading Scaler model:", e)
    scalar = None

try:
    meta_cat_model = joblib.load("models/meta_cat_model1.pkl")
    print("Meta CatBoost model loaded successfully.")
except Exception as e:
    print("Error loading Meta CatBoost model:", e)
    meta_cat_model = None

print("All models loaded!")

# -------------------------------
# Helper: WQI Classification
# -------------------------------
def classify_wqi(wqi):
    if wqi >= 90:
        return "Excellent"
    elif wqi >= 70:
        return "Good"
    elif wqi >= 50:
        return "Moderate"
    else:
        return "Poor"

# -------------------------------
# Root Endpoint
# -------------------------------
@app.route('/')
def home():
    return jsonify({"message": "AquaNet-X Backend is running successfully!"})

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        # Extract inputs from frontend
        pH = float(data.get('pH', 0))
        DO = float(data.get('DO', 0))
        TEMP = float(data.get('TEMP', 0))
        BOD = float(data.get('BOD', 0))
        FS = float(data.get('FS', 0))
        TC = float(data.get('TC', 0))
        FC = float(data.get('FC', 0))
        COND = float(data.get('COND', 0))
        NO3 = float(data.get('NO3', 0))
        
        
        features = np.array([[pH, DO, TEMP, BOD, FS, TC, FC, COND, NO3]], dtype=np.float32)
        # gru_input = features.reshape((1, 1, features.shape[1]))
        print("Input features:", features) 
        from sklearn.preprocessing import StandardScaler
        scaled_input = scaler.transform(features)
        print("sccaled features:", scaled_input)
        
        
        # reshape for GRU/Transformer input (1 sample, 1 timestep, 9 features)
        gru_input = scaled_input.reshape((1, 1, 9))

        predictions = []

        # Predict from available models
        if gru_model:
            gru_pred = float(gru_model.predict(gru_input, verbose=0)[0][0])
            # gru_pred = float(gru_model.predict(gru_input, verbose=0).flatten()[0])
            predictions.append(gru_pred)
            print("GRU prediction:", gru_pred)
        if transformer_model:
            trans_pred = float(transformer_model.predict(gru_input, verbose=0)[0][0])
            # trans_pred = float(transformer_model.predict(gru_input, verbose=0).flatten()[0])
            predictions.append(trans_pred)
            print("Transformer prediction:", trans_pred)
        if xgb_model:
            xgb_pred = float(xgb_model.predict(scaled_input)[0])
            # xgb_pred = float(xgb_model.predict(scaled_input).flatten()[0])
            predictions.append(xgb_pred)
            print("XGBoost prediction:", xgb_pred)
            # preditions.append(xgb_model.predict(features)[0])
        if lgb_model:
            lgb_pred = float(lgb_model.predict(scaled_input)[0])
            predictions.append(lgb_pred)
            print("LightGBM prediction:", lgb_pred)
            # predictions.append(lgb_model.predict(features)[0])

        # if len(predictions) == 0:
        if not predictions:
            return jsonify({"error": "No base models loaded!"})

        # Meta-model fusion
        avg_pred = np.array([[gru_pred, xgb_pred, trans_pred, lgb_pred]])

        # final_pred = meta_cat_model.predict(avg_pred)[0] if meta_cat_model else avg_pred[0][0]
        # category = classify_wqi(final_pred)

        
        
        if meta_cat_model:
            final_pred = meta_cat_model.predict(avg_pred)[0]
        else:
            final_pred = avg_pred[0][0]
        
        if ( final_pred >= 100):
            final=99.999
        elif ( final_pred <= 0):
            final = 0.00
        else :
            final=final_pred
            
        print(f"Final Meta prediction:{final:.7f}")

        category = classify_wqi(final)
        # print(f"Predicted WQI: {final_pred:.2f}, Category: {category}")

        return jsonify({
            "Predicted_WQI": round(final, 5),
            "Category": category
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)})

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  
# debug=True


# backend run = python app.py
# frontend run = npm start

# prediction=84.64382
# pH = 6.5
# DO = 4.5
# Temp = 29
# BOD = 7.1
# FS = 35
# TC = 220
# FC = 150
# Cond = 250
# NO3 = 18
# pH=7.3, DO=8.5, Temp=25, BOD=2.8, FS=10, TC=85, FC=60, Cond=160, NO3=7
# ➡ Predicted WQI ≈ 85
# pH=7.0, DO=6.8, Temp=27, BOD=4.2, FS=20, TC=140, FC=110, Cond=180, NO3=12
# ➡ Predicted WQI ≈ 70
# pH=6.9, DO=5.2, Temp=28, BOD=6.1, FS=25, TC=180, FC=120, Cond=220, NO3=14
# ➡ Predicted WQI ≈ 58
# pH=6.4, DO=4.1, Temp=30, BOD=8.5, FS=45, TC=250, FC=190, Cond=280, NO3=20
# ➡ Predicted WQI ≈ 42
# pH=7.5, DO=9.0, Temp=24, BOD=2.0, FS=5, TC=50, FC=30, Cond=120, NO3=5
# ➡ Predicted WQI ≈ 92
# pH=6.2, DO=3.8, Temp=31, BOD=9.2, FS=55, TC=290, FC=220, Cond=330, NO3=25
# ➡ Predicted WQI ≈ 36
# pH=7.1, DO=6.0, Temp=26, BOD=5.0, FS=22, TC=160, FC=130, Cond=200, NO3=16
# ➡ Predicted WQI ≈ 63
# pH=6.7, DO=4.8, Temp=29, BOD=7.4, FS=32, TC=210, FC=170, Cond=260, NO3=22
# ➡ Predicted WQI ≈ 48
# pH=7.0, DO=7.4, Temp=26, BOD=3.8, FS=15, TC=120, FC=85, Cond=170, NO3=9
# ➡ Predicted WQI ≈ 75
# pH=6.0, DO=2.9, Temp=32, BOD=11.0, FS=70, TC=350, FC=300, Cond=400, NO3=30
# ➡ Predicted WQI ≈ 28