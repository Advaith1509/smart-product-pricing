# src/predict.py

import pandas as pd
import numpy as np
import joblib
import config
from train import load_features # Re-use the feature loading logic

def main():
    """Generates predictions on the test set and creates a submission file."""
    print("Loading trained models...")
    models = joblib.load(os.path.join(config.MODEL_DIR, 'lgbm_models.pkl'))
    
    X_test, _ = load_features('test')
    
    print("Generating predictions on the test set...")
    test_preds = np.zeros(X_test.shape)
    for model in models:
        # Sum predictions from each fold model and average later
        test_preds += np.expm1(model.predict(X_test)) / len(models)
        
    # Create submission file
    test_df = pd.read_csv(config.TEST_CSV)
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    # Ensure prices are positive
    submission_df['price'] = submission_df['price'].clip(lower=0)
    
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSION_DIR, 'final_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file created successfully at: {submission_path}")
    print(submission_df.head())

if __name__ == "__main__":
    main()