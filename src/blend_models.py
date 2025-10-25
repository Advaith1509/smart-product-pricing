# src/blend_models.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import os
import config

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    This function assumes y_true and y_pred are already inverse-transformed (i.e., on the original price scale).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Add a small epsilon to the denominator to avoid division by zero
    return np.mean(numerator / (denominator + 1e-6)) * 100

def load_oof_and_test_predictions(model_files):
    """
    Loads out-of-fold (OOF) and test predictions from specified files.
    
    Args:
        model_files (dict): A dictionary mapping model names to their prediction file paths.
                            Each file should contain 'sample_id' and the model's prediction column.

    Returns:
        tuple: A tuple containing:
            - oof_df (pd.DataFrame): Merged OOF predictions.
            - test_df (pd.DataFrame): Merged test predictions.
    """
    oof_dfs = []
    test_dfs = []
    
    print("Loading prediction files...")
    lgbm_preds = pd.read_csv(config.SUBMISSION_DIR + "submission.csv")
    transformer_preds = pd.read_csv(config.SUBMISSION_DIR + "pytorch_transformer_submission.csv")
    
    test_df = pd.merge(lgbm_preds.rename(columns={'price': 'lgbm'}),
                       transformer_preds.rename(columns={'price': 'transformer'}),
                       on='sample_id')

    print("WARNING: Using placeholder OOF data. Replace with your actual OOF files for a valid result.")
    train_df = pd.read_csv(config.TRAIN_CSV)
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price': train_df['price'],
        'lgbm': np.random.rand(len(train_df)) * 100, # Replace with real OOF preds
        'transformer': np.random.rand(len(train_df)) * 100 # Replace with real OOF preds
    })
    
    return oof_df, test_df


def optimize_weights(oof_df):
    """
    Finds the optimal weights for blending model predictions using binary search-like optimization.
    """
    predictions = oof_df.drop(columns=['sample_id', 'price']).values
    y_true = oof_df['price'].values
    
    def objective_function(weights):
        """The function to minimize (SMAPE score)."""
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        blended_preds = np.sum(predictions * weights, axis=1)
        return smape(y_true, blended_preds)

    print("Optimizing ensemble weights...")
    # Initial guess: equal weights
    initial_weights = np.ones(predictions.shape[1]) / predictions.shape[1]
    
    # Constraint: sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
    
    # Bounds: each weight must be between 0 and 1
    bounds = [(0, 1)] * len(initial_weights)
    
    result = minimize(objective_function,
                      initial_weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    optimal_weights = result.x
    min_smape = result.fun
    
    print(f"Optimization complete. Minimum SMAPE found: {min_smape:.4f}")
    print(f"Optimal Weights: {optimal_weights}")
    
    return optimal_weights

def main():
    """Main function to blend predictions and create the final submission."""
    oof_df, test_df = load_oof_and_test_predictions()
    
    # Find the best weights
    optimal_weights = optimize_weights(oof_df)
    
    # Apply weights to the test predictions
    model_cols = [col for col in test_df.columns if col!= 'sample_id']
    test_predictions = test_df[model_cols].values
    
    blended_predictions = np.sum(test_predictions * optimal_weights, axis=1)
    
    # Create final submission file
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': blended_predictions
    })
    
    # Ensure prices are positive
    submission_df['price'] = submission_df['price'].clip(lower=0)
    
    final_submission_path = os.path.join(config.SUBMISSION_DIR, 'ensemble_submission.csv')
    submission_df.to_csv(final_submission_path, index=False)
    
    print(f"\nFinal blended submission file created at: {final_submission_path}")
    print(submission_df.head())

if __name__ == "__main__":
    main()