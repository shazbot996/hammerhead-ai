#!/bin/bash

# This script stops the Hammerhead AI systemd service.

echo "Stopping the Hammerhead AI service..."
sudo systemctl stop hammerhead-ai.service
echo "Service stopped."