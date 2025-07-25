#!/bin/bash

echo "Warning! This script downloads a lot of reanalysis data, and it can take up to 12 hours to run."
echo "Please keep this in mind. If you have synthetic data, or are just interested in the theoretical results from the paper, you can just run those figure scripts without downloading the data."

python data_mains/download_era5.py
python data_mains/process_era5.py

echo "Done!"
