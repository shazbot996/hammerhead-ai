#!/bin/bash

# update_config_on_sd.sh
# This script updates the config.json on the SD card with the local version from this project.
# It must be run with sudo privileges from the project's root directory.
#
# Usage:
#   sudo ./update_config_on_sd.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SD_CARD_DEVICE="/dev/mmcblk1p1"
MOUNT_POINT="/mnt/sdcard"
SOURCE_CONFIG_FILE="local_data_cache/config/config.json"
DEST_DATA_DIR="${MOUNT_POINT}/hammerhead_data"
DEST_CONFIG_DIR="${DEST_DATA_DIR}/config"
DEST_CONFIG_FILE="${DEST_CONFIG_DIR}/config.json"

# --- Pre-flight Checks ---

# 1. Check if running as root (sudo)
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: This script must be run as root. Please use 'sudo'."
  exit 1
fi

# 2. Check if the local config file exists
if [ ! -f "$SOURCE_CONFIG_FILE" ]; then
  echo "ERROR: Local config file not found at '$SOURCE_CONFIG_FILE'."
  echo "Please ensure you are running this script from the project's root directory."
  exit 1
fi

# 3. Check if the SD card device is present
if [ ! -b "$SD_CARD_DEVICE" ]; then
    echo "ERROR: SD card device ($SD_CARD_DEVICE) not found. Is the card inserted?"
    exit 1
fi

# --- Main Logic ---

# Define a cleanup function to unmount the card on exit or error
cleanup() {
  echo "--- Running cleanup ---"
  # Check if the mount point is actually mounted before trying to unmount
  if mountpoint -q "$MOUNT_POINT"; then
    echo "Unmounting $MOUNT_POINT..."
    umount "$MOUNT_POINT"
  fi
  echo "Cleanup complete."
}

# Register the cleanup function to run on script exit (normal or error)
trap cleanup EXIT

echo "--- Starting SD Card Config Update ---"

# 1. Mount the SD card
echo "Creating mount point at $MOUNT_POINT (if it doesn't exist)..."
mkdir -p "$MOUNT_POINT"
echo "Mounting $SD_CARD_DEVICE to $MOUNT_POINT..."
mount "$SD_CARD_DEVICE" "$MOUNT_POINT"
echo "Mount successful."

# 2. Copy the configuration file
echo "Ensuring destination directory exists: $DEST_CONFIG_DIR"
mkdir -p "$DEST_CONFIG_DIR"

echo "Copying local config to SD card..."
echo "  Source:      $SOURCE_CONFIG_FILE"
echo "  Destination: $DEST_CONFIG_FILE"
cp -v "$SOURCE_CONFIG_FILE" "$DEST_CONFIG_FILE"

echo "--- SUCCESS: Configuration file has been updated on the SD card. ---"

# The 'trap' will handle unmounting automatically when the script exits here.

