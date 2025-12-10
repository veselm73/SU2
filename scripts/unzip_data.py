import zipfile
import os

zip_path = "val_and_sota.zip"
extract_to = "val_data"

if os.path.exists(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is a bad zip file. It might be incomplete.")
else:
    print(f"Error: {zip_path} does not exist.")
