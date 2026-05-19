import subprocess
import tempfile
import os
import shutil
from tqdm import tqdm
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Repository URL
repo_url = "https://github.com/camaramm/tennessee-eastman-profBraatz.git"
raw_dir = "chembench/datasets/tennessee_eastman/raw/"

# File mappings
fault_free_training_file = "d00.dat"
fault_free_testing_file = "d00_te.dat"
faulty_training_files = [f"d{i:02d}.dat" for i in range(1, 22)]  # d01.dat to d21.dat
faulty_testing_files = [f"d{i:02d}_te.dat" for i in range(1, 22)]  # d01_te.dat to d21_te.dat

def clone_repo(repo_url, temp_dir):
    """Clone the repository to a temporary directory."""
    try:
        logging.info("Cloning repository...")
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True)
        logging.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to clone repository: {e}")

def read_dat_file(file_path):
    """Read a .dat file into a pandas DataFrame."""
    try:
        # Assuming space-separated values, no header
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        return df
    except Exception as e:
        raise Exception(f"Failed to read {file_path}: {e}")

def combine_files(temp_dir, files):
    """Combine multiple .dat files into a single DataFrame."""
    dfs = []
    for file in tqdm(files, desc="Reading files"):
        file_path = os.path.join(temp_dir, file)
        if os.path.exists(file_path):
            df = read_dat_file(file_path)
            dfs.append(df)
        else:
            logging.warning(f"File {file} not found.")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        raise Exception("No files found to combine.")

def save_csv(df, csv_path):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved {csv_path}")
    except Exception as e:
        raise Exception(f"Failed to save {csv_path}: {e}")

def main():
    # Ensure raw directory exists
    os.makedirs(raw_dir, exist_ok=True)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repository
        clone_repo(repo_url, temp_dir)

        # Process fault-free training
        ff_train_path = os.path.join(temp_dir, fault_free_training_file)
        if os.path.exists(ff_train_path):
            df_ff_train = read_dat_file(ff_train_path)
            save_csv(df_ff_train, os.path.join(raw_dir, "fault_free_training.csv"))
        else:
            logging.error(f"{fault_free_training_file} not found.")

        # Process fault-free testing
        ff_test_path = os.path.join(temp_dir, fault_free_testing_file)
        if os.path.exists(ff_test_path):
            df_ff_test = read_dat_file(ff_test_path)
            save_csv(df_ff_test, os.path.join(raw_dir, "fault_free_testing.csv"))
        else:
            logging.error(f"{fault_free_testing_file} not found.")

        # Process faulty training
        df_faulty_train = combine_files(temp_dir, faulty_training_files)
        save_csv(df_faulty_train, os.path.join(raw_dir, "faulty_training.csv"))

        # Process faulty testing
        df_faulty_test = combine_files(temp_dir, faulty_testing_files)
        save_csv(df_faulty_test, os.path.join(raw_dir, "faulty_testing.csv"))

    logging.info("Dataset download and conversion completed successfully.")

if __name__ == "__main__":
    main()