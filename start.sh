#!/bin/bash

# This script starts the Hammerhead AI facial recognition application.

# Get the absolute path of the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script's directory to ensure all relative paths in the Python script work correctly.
cd "$SCRIPT_DIR"

echo "Starting Hammerhead AI application..."


# The line below is a WORKAROUND for a dependency conflict between older libraries
# (like face_recognition or edgetpu) and newer Google Cloud libraries.
# It forces protobuf to use a slower, pure-Python implementation that is more
# compatible across different versions. This allows us to use a modern protobuf
# version required by google-cloud-* packages while still running the older code.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
 
DISPLAY=:0 python3 main.py
