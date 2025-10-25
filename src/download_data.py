# src/download_data.py

import os
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import config

def download_single_image(url, filepath, max_retries=3, timeout=10):
    """Downloads a single image with retry logic."""
    if os.path.exists(filepath):
        return True
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=timeout, headers=headers)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                print(f"Failed to download {url}: {e}")
    return False

def download_image_batch(df, image_dir, id_col='sample_id', url_col='image_link'):
    """Downloads a batch of images in parallel."""
    os.makedirs(image_dir, exist_ok=True)
    
    tasks = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for _, row in df.iterrows():
            sample_id = row[id_col]
            url = row[url_col]
            # Assumes image names are sample_id +.jpg
            filepath = os.path.join(image_dir, f"{sample_id}.jpg")
            tasks.append(executor.submit(download_single_image, url, filepath))
            
        for future in tqdm(as_completed(tasks), total=len(tasks), desc=f"Downloading to {os.path.basename(image_dir)}"):
            future.result()

def main():
    """Main function to download all training and testing images."""
    print("Loading datasets...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    
    print("Starting training image download...")
    download_image_batch(train_df, config.TRAIN_IMAGE_DIR)
    
    print("\nStarting testing image download...")
    download_image_batch(test_df, config.TEST_IMAGE_DIR)
    
    print("\nImage download process complete.")

if __name__ == "__main__":
    main()