"""Downloads the Sleep-EDF (cassette) dataset from Kaggle into data/raw/.

Requires a Kaggle API token. One-time setup:
  1. Go to kaggle.com -> Settings -> API -> "Create New Token".
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json (chmod 600),
     or set KAGGLE_USERNAME / KAGGLE_KEY environment variables instead.

Usage:
  python download_data.py
"""

import os
import sys

import config


def main():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "The 'kaggle' package isn't installed. Run:\n"
            "  pip install kaggle\n"
            "then re-run this script."
        )

    token_path = os.path.expanduser("~/.kaggle/kaggle.json")
    has_env_creds = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not os.path.exists(token_path) and not has_env_creds:
        sys.exit(
            "No Kaggle credentials found.\n"
            f"Expected a token at {token_path}, or KAGGLE_USERNAME/KAGGLE_KEY "
            "environment variables.\n"
            "Create one at https://www.kaggle.com/settings -> API -> "
            "'Create New Token', then re-run this script."
        )

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {config.KAGGLE_DATASET} -> {config.RAW_DIR} ...")
    api.dataset_download_files(
        config.KAGGLE_DATASET, path=config.RAW_DIR, unzip=True, quiet=False
    )
    print("Done. PSG/Hypnogram EDF files should now be under:", config.RAW_DIR)


if __name__ == "__main__":
    main()
