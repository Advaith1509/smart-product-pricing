# src/train.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
import scipy.sparse
import joblib
import config

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error metric."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def load_features(dataset_type='train'):
    """Loads and merges all features for a given dataset type (train/test)."""
    print(f"Loading features for {dataset_type} set...")
    
    # Load base dataframe
    df_path = config.TRAIN_CSV if dataset_type == 'train' else config.TEST_CSV
    df = pd.read_csv(df_path)
    
    # Load image features and merge
    for model_name in config.MODEL_CONFIGS.keys():
        img_feat_path = os.path.join(config.FEATURE_DIR, f'{model_name}_{dataset_type}_features.parquet')
        img_df = pd.read_parquet(img_feat_path)
        df = pd.merge(df, img_df, on='sample_id', how='left')
        
    # Load text features
    text_feat_path = os.path.join(config.FEATURE_DIR, f'{dataset_type}_text_features.parquet')
    text_df = pd.read_parquet(text_feat_path)
    df = pd.merge(df, text_df, on='sample_id', how='left')
    
    tfidf_path = os.path.join(config.FEATURE_DIR, f'{dataset_type}_tfidf.npz')
    tfidf_features = scipy.sparse.load_npz(tfidf_path)
    
    transformer_path = os.path.join(config.FEATURE_DIR, f'{dataset_type}_transformer_embeddings.npy')
    transformer_features = np.load(transformer_path)
    
    # Combine all features
    feature_cols = [c for c in df.columns if 'feat' in c or c == 'ipq']
    final_features = scipy.sparse.hstack([
        df[feature_cols].values,
        tfidf_features,
        transformer_features
    ]).tocsr()
    
    return final_features, df if dataset_type == 'train' else None

def main():
    """Main function to train the model."""
    X, y = load_features('train')
    
    # Log transform the target variable
    y_log = np.log1p(y)
    
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_preds = np.zeros(X.shape)
    models =
    
    print("\nStarting model training with 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"--- Fold {fold+1}/{config.N_SPLITS} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        models.append(model)
        
    # Evaluate OOF predictions
    oof_smape = smape(y, np.expm1(oof_preds))
    print(f"\nOverall Out-of-Fold SMAPE: {oof_smape:.4f}")
    
    # Save the trained models
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_models.pkl')
    joblib.dump(models, model_path)
    print(f"Models saved to {model_path}")

if __name__ == "__main__":
    main()