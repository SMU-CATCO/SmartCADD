#!/bin/bash

# Exit on error
set -e

# Initialize variable for named parameters
config_file=""
experiment_name=""
package_name=""

# Loop through arguments and process them
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file) config_file="$2"; shift ;;
        --experiment_name) experiment_name="$2"; shift ;;
        --package_name) package_name="$3"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Move to next argument
done

# Logging for debugging
echo "==========================="
echo $(ls)
echo "==========================="

# Install poetry
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry --force

# Install dependencies (superpod doesn't like poetry)
if [ -f poetry.lock ]; then
    rm -r poetry.lock
fi
poetry export -f requirements.txt --without-hashes --output requirements.txt
pip install -r requirements.txt
rm -f requirements.txt
if [ -f poetry.toml ]; then
    rm -r poetry.toml
fi

# Install our custom package
pip install -e .

# CD to the experiment directory
cd experiments/${experiment_name}

# Logging for debugging
echo "==========================="
echo "Should be in the experiment directory"
echo $(ls)
echo "==========================="

# Run the experiment
python -m bocas.launch run.py --task run.py --config configs/"${config_file}"