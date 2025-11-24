import requests
import zipfile
import os

url = "https://su2.utia.cas.cz/files/labs/final2025/val_and_sota.zip"
zip_path = "val_data_v2.zip"
extract_to = "val_data"

print(f"Downloading {url} to {zip_path}...")
try:
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
    
    print(f"Extracting to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")
    
except Exception as e:
    print(f"Error: {e}")
