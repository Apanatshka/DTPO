#!/usr/bin/env bash

# Set CWD as where this script is
cd "$(dirname "$0")" || exit

# Load the virtual environment
source .venv/bin/activate

env

# run_dqn through UV to use venv, forwarding the shell arguments
python src/dtpo/run_dqn.py "$@"
