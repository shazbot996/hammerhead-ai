#!/bin/bash

# This script acts as a centralized, passwordless entry point for all operations
# that require root privileges (sudo). It is intended to be called by the main
# application and configured in /etc/sudoers for passwordless execution.

# Exit immediately if a command exits with a non-zero status.
set -e

COMMAND="$1"

SD_CARD_DEVICE="/dev/mmcblk1p1"
MOUNT_POINT="/mnt/sdcard"
SOURCE_DATA_DIR="hammerhead_data"
LOCAL_CACHE_DIR="/home/mendel/hammerhead-ai/local_data_cache"

case "$COMMAND" in
    mount_sd)
        echo "HELPER: Mounting SD card..."
        mkdir -p "$MOUNT_POINT"
        mount "$SD_CARD_DEVICE" "$MOUNT_POINT"
        ;;

    umount_sd)
        echo "HELPER: Unmounting SD card..."
        # Check if it's actually mounted before trying to unmount
        if mountpoint -q "$MOUNT_POINT"; then
            umount "$MOUNT_POINT"
        else
            echo "HELPER: Mount point not active. Skipping unmount."
        fi
        ;;

    sync_data)
        echo "HELPER: Syncing data from SD card..."
        SD_DATA_ROOT="${MOUNT_POINT}/${SOURCE_DATA_DIR}"

        # Sync face-index directory
        SD_FACES_DIR="${SD_DATA_ROOT}/face-index"
        LOCAL_FACES_DIR="${LOCAL_CACHE_DIR}/face-index"
        if [ -d "$SD_FACES_DIR" ]; then
            echo "HELPER: Syncing face-index directory..."
            mkdir -p "$LOCAL_FACES_DIR"
            # Use 'cp -r' to copy contents, not 'cp -a' which preserves root ownership from SD card.
            cp -r "${SD_FACES_DIR}/." "$LOCAL_FACES_DIR/"
            chown -R mendel:mendel "$LOCAL_FACES_DIR"
        else
            echo "HELPER: 'face-index' directory not found on SD card. Skipping face sync."
        fi

        # Sync config.json file
        SD_CONFIG_FILE_PATH="${SD_DATA_ROOT}/config/config.json"
        LOCAL_CONFIG_DIR="${LOCAL_CACHE_DIR}/config" # e.g., .../local_data_cache/config
        LOCAL_CONFIG_FILE_PATH="${LOCAL_CONFIG_DIR}/config.json" # e.g., .../local_data_cache/config/config.json
        if [ -f "$SD_CONFIG_FILE_PATH" ]; then
            echo "HELPER: Found config.json at '${SD_CONFIG_FILE_PATH}'. Syncing..."
            mkdir -p "$LOCAL_CONFIG_DIR"
            # Use a simple 'cp' to copy the file.
            cp "$SD_CONFIG_FILE_PATH" "$LOCAL_CONFIG_FILE_PATH"
            # Set ownership on the specific file that was copied.
            chown mendel:mendel "$LOCAL_CONFIG_FILE_PATH"
        else
            echo "HELPER: config.json not found at '${SD_CONFIG_FILE_PATH}'. Skipping config sync."
        fi
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'" >&2
        exit 1
        ;;
esac

echo "HELPER: Command '$COMMAND' completed successfully."