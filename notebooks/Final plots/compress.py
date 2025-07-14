import os
import torch
import pickle
import gzip

import os
import torch
import pickle
import gzip

def compress_to_pickle(file_path):
    try:
        # Load data depending on file type
        if file_path.endswith('.pt'):
            with open(file_path, 'rb') as f:
                data = torch.load(f)

        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            return

        # Create consistent .pkl.gz filename
        compressed_path = file_path.rsplit('.', 1)[0] + '.pkl.gz'

        # Save using pickle with gzip
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(data, f)

        # Remove original after successful save
        os.remove(file_path)
        print(f"✅ Compressed & removed: {file_path} -> {compressed_path}")

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

def compress_all_files(root_dir):
    for folder, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(folder, file)

            # Skip already compressed files
            if file.endswith('.gz'):
                continue

            if file.endswith('.pt') or file.endswith('.pkl'):
                compress_to_pickle(file_path)


if __name__ == "__main__":
    root_directory = "/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/ETTm1"  # Change this to your root folder
    compress_all_files(root_directory)
