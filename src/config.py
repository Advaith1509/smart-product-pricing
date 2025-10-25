# src/config.py

# --- Project Paths ---
DATA_DIR = "dataset/"
TRAIN_IMAGE_DIR = "dataset/train_images/"
TEST_IMAGE_DIR = "dataset/test_images/"
FEATURE_DIR = "features/"
MODEL_DIR = "models/"
SUBMISSION_DIR = "submissions/"
INDIVIDUAL_PREDICTIONS_DIR = "predicted_values_csv/" # Directory for base model predictions

# --- Data Files ---
TRAIN_CSV = f"{DATA_DIR}train.csv"
TEST_CSV = f"{DATA_DIR}test.csv"

# --- Feature Engineering ---
# Image Models
MODEL_CONFIGS = {
    'efficientnet_b2': {
        'img_size': 260,
    },
    'convnext_tiny': {
        'img_size': 224,
    }
}

# Text Models
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2

# --- Training Parameters ---
TARGET_COL = "price"
N_SPLITS = 5
RANDOM_STATE = 42

# LightGBM Parameters
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',
}


ENSEMBLE_MODEL_FILES = {
    'lgbm': {
        'oof': f"{INDIVIDUAL_PREDICTIONS_DIR}lgbm_oof.csv", # You must save OOF preds here
        'test': f"{INDIVIDUAL_PREDICTIONS_DIR}submission.csv"
    },
    'transformer': {
        'oof': f"{INDIVIDUAL_PREDICTIONS_DIR}transformer_oof.csv", # You must save OOF preds here
        'test': f"{INDIVIDUAL_PREDICTIONS_DIR}pytorch_transformer_submission.csv"
    }
}

# Path for the final, optimized blended submission file
FINAL_SUBMISSION_FILE = f"{SUBMISSION_DIR}final_ensemble_submission.csv"