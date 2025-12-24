import os
import sys

# Read DATA_ROOT from environment variable (set in config/run_config.sh)
DATA_ROOT = os.getenv("DATA_ROOT")

if DATA_ROOT is None:
    # Fallback or Error if not set
    print("Warning: DATA_ROOT environment variable not set. Please ensure config/run_config.sh is sourced.")
    # Defaulting to a placeholder to prevent immediate import crashes, but runtime errors will occur if not fixed.
    DATA_ROOT = "/hkfs/home/project/hk-project-p0024638/uzivy/datasets"
