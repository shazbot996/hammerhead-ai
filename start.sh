#!/bin/bash

# This script starts the Hammerhead AI facial recognition application.
# It can be run in two modes:
#   - With GUI: ./start.sh
#   - Headless (no GUI): ./start.sh --headless

# Get the absolute path of the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the pixel format for V4L2 to prevent OpenCV errors with certain cameras (e.g., MJPEG default).
# 'YUYV' is a widely supported format that OpenCV can handle.
export V4L2_PIXEL_FORMAT=YUYV

# --- Main Logic ---

# Default to running with the display
FIRST_ARG="$1"

# Check for command-line arguments
if [ "$FIRST_ARG" == "--headless" ]; then
    echo "Starting Hammerhead AI application in HEADLESS mode..."
    # Run without a display. Pass the --headless argument to the Python script.
    # Your Python script (main.py) must be updated to handle this argument
    # and avoid creating any GUI windows (e.g., cv2.imshow()).
    python3 "${SCRIPT_DIR}/main.py" "$@"
elif [ "$FIRST_ARG" == "--stream" ]; then
    echo "Starting Hammerhead AI application in STREAMING mode..."
    # Run in headless mode and start the web stream.
    # The Python script will handle starting the server.
    python3 "${SCRIPT_DIR}/main.py" "$@"
else
    # This is the default case (GUI mode, or any other combination of flags like --force-index)
    echo "Starting Hammerhead AI application with GUI..."
    # The `DISPLAY=:0` part tells the application to use the primary display.
    DISPLAY=:0 python3 "${SCRIPT_DIR}/main.py" "$@"
fi
