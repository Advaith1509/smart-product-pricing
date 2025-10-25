# src/feature_engineering.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import config

# --- Image Feature Extraction ---

class ProductImageDataset(Dataset):
    """PyTorch Dataset for loading product images."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, filename
        except Exception:
            # Return a placeholder if image is corrupt
            return torch.zeros((3, 224, 224)), filename

def get_model_and_transforms(model_name):
    """Loads a pre-trained model and its corresponding transforms."""
    if model_name == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.DEFAULT
        model = models.efficientnet_b2(weights=weights)
        model.classifier = torch.nn.Identity()
    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    transforms = weights.transforms()
    return model, transforms

def extract_image_features(model_name, image_dir, output_path):
    """Extracts and saves image features using a specified model."""
    print(f"--- Starting image feature extraction for '{model_name}' ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_transforms = get_model_and_transforms(model_name)
    model.to(device)
    model.eval()

    dataset = ProductImageDataset(image_dir, transform=model_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    all_features = {}
    with torch.no_grad():
        for batch_images, batch_files in tqdm(dataloader, desc=f"Extracting {model_name}"):
            input_tensor = batch_images.to(device)
            features = model(input_tensor)
            features_cpu = features.cpu().numpy()
            
            for i, filename in enumerate(batch_files):
                sample_id = os.path.splitext(filename)
                all_features[sample_id] = features_cpu[i]

    features_df = pd.DataFrame.from_dict(all_features, orient='index')
    features_df.columns = [f"{model_name}_feat_{i}" for i in range(features_df.shape[1])]
    features_df.index.name = 'sample_id'
    features_df.reset_index(inplace=True)
    features_df.to_parquet(output_path, index=False)
    print(f"Image features saved to {output_path}")

# --- Text Feature Engineering ---

def extract_text_features(df, text_col='catalog_content'):
    """Extracts IPQ, TF-IDF, and Transformer embeddings from text."""
    print("--- Starting text feature extraction ---")
    
    # 1. Extract IPQ (Item Pack Quantity)
    df['ipq'] = df[text_col].str.extract(r'(?:pack of|pk of|pack|count)\s*(\d+)', expand=False).fillna(1).astype(int)
    
    # 2. TF-IDF Features
    print("Generating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(df[text_col].fillna(""))
    
    # 3. Transformer Embeddings
    print("Generating Transformer embeddings...")
    model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')
    transformer_embeddings = model.encode(df[text_col].fillna("").tolist(), show_progress_bar=True, batch_size=128)
    
    return df[['sample_id', 'ipq']], tfidf_features, transformer_embeddings

def main():
    """Main function to run all feature engineering steps."""
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    
    # --- Image Features ---
    for model_name in config.MODEL_CONFIGS.keys():
        # Process Training Images
        train_output_path = os.path.join(config.FEATURE_DIR, f'{model_name}_train_features.parquet')
        extract_image_features(model_name, config.TRAIN_IMAGE_DIR, train_output_path)
        
        # Process Testing Images
        test_output_path = os.path.join(config.FEATURE_DIR, f'{model_name}_test_features.parquet')
        extract_image_features(model_name, config.TEST_IMAGE_DIR, test_output_path)

    # --- Text Features ---
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    
    train_text_df, train_tfidf, train_transformer = extract_text_features(train_df)
    test_text_df, test_tfidf, test_transformer = extract_text_features(test_df)
    
    # Save text features
    train_text_df.to_parquet(f"{config.FEATURE_DIR}train_text_features.parquet", index=False)
    test_text_df.to_parquet(f"{config.FEATURE_DIR}test_text_features.parquet", index=False)
    scipy.sparse.save_npz(f"{config.FEATURE_DIR}train_tfidf.npz", train_tfidf)
    scipy.sparse.save_npz(f"{config.FEATURE_DIR}test_tfidf.npz", test_tfidf)
    np.save(f"{config.FEATURE_DIR}train_transformer_embeddings.npy", train_transformer)
    np.save(f"{config.FEATURE_DIR}test_transformer_embeddings.npy", test_transformer)
    
    print("\nAll feature engineering tasks are complete.")

if __name__ == "__main__":
    main()