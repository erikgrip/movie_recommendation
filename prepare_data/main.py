""" Main script to prepare data for the project. """

import subprocess


def run_pipeline():
    """Runs the data preparation pipeline."""
    subprocess.run(["poetry", "run", "python", "prepare_data/download.py"], check=True)
    subprocess.run(["poetry", "run", "python", "prepare_data/clean.py"], check=True)
    subprocess.run(["poetry", "run", "python", "prepare_data/features.py"], check=True)


if __name__ == "__main__":
    run_pipeline()
