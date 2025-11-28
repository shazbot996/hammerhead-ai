#!/bin/bash

# This script starts the Hammerhead AI facial recognition application.

# Get the absolute path of the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script's directory to ensure all relative paths in the Python script work correctly.
# cd "$SCRIPT_DIR" # Temporarily disabled for troubleshooting. We will call python with an absolute path.

echo "Starting Hammerhead AI application..."

# Set the pixel format for V4L2 to prevent OpenCV errors with certain cameras (e.g., MJPEG default).
# 'YUYV' is a widely supported format that OpenCV can handle.
export V4L2_PIXEL_FORMAT=YUYV

# The `DISPLAY=:0` part tells the application to use the primary display connected to the board.
# We now use an absolute path to the main script to avoid any issues related to the current working directory.
# Explicitly calling 'python3' is more robust and avoids issues with file line-endings (CRLF vs LF).
DISPLAY=:0 python3 "${SCRIPT_DIR}/main.py"
