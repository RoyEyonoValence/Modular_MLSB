#!/usr/bin/env bash --login
set -e

ANACONDA_USER_TOKEN=$1

if [[ -z $ANACONDA_USER_TOKEN ]]; then
  echo 'Error: one or more credentials variables are not set.'
  exit 1
fi

if [[ -z $CONDA_ENV_NAME ]]; then
  echo 'Error: CONDA_ENV_NAME must be set.'
  exit 1
fi

cd /app

# Login to Anaconda
# Bypass the installation of anaconda-client
TOKEN_DIR="$HOME/.config/binstar"
TOKEN_PATH="$TOKEN_DIR/https%3A%2F%2Fapi.anaconda.org.token"
mkdir -p $TOKEN_DIR
echo -e "${ANACONDA_USER_TOKEN}\c" >$TOKEN_PATH

# Create conda env
mamba env create --name $CONDA_ENV_NAME -f env.yml

# Activate the env
conda activate $CONDA_ENV_NAME

# Install your app
pip install -e .

# Clean up space
conda clean --all --yes

# Force conda env activation
echo "conda activate $CONDA_ENV_NAME" >>"$HOME/.bashrc"
