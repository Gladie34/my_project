import joblib

# Load saved model and preprocessing tools
MODEL_PATH = "saved_xgb_model.pkl"
ENCODER_SCALER_PATH = "encoder_scaler.pkl"

# Load assets once
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    preprocessing_assets = joblib.load(ENCODER_SCALER_PATH)
    encoders = preprocessing_assets['encoders']
    scaler = preprocessing_assets['scaler']
    feature_names = preprocessing_assets['features']
    return model, encoders, scaler, feature_names

# Define appropriate ranges for numeric features
NUMERIC_RANGES = {
    # Default range for unspecified numeric features
    "default": (0.0, 100.0, 1.0),
    
    # Custom ranges for specific features - update these based on your actual features
    "Income": (0.0, 250000.0, 5000.0),
    "Age": (18.0, 90.0, 1.0),
    "ClaimAmount": (100.0, 100000.0, 100.0),
    "PolicyDuration": (0.0, 30.0, 0.5),
    "PremiumAmount": (100.0, 10000.0, 100.0),
    "DaysToFile": (0.0, 365.0, 1.0),
    "ClaimComplexityScore": (1.0, 10.0, 0.1)
}

def get_range_for_feature(feature_name):
    """Returns appropriate min, max, and step values for a numeric feature"""
    if feature_name in NUMERIC_RANGES:
        return NUMERIC_RANGES[feature_name]
    else:
        return NUMERIC_RANGES["default"]