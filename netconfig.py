#!/usr/bin/env python3
# netconfig.py

import os
import json
import subprocess
import time

# The fixed location where the network configuration file is expected on the mounted SD card.
NETCONFIG_FILE_PATH = "/mnt/sdcard/netconfig.json"

def get_current_wifi_ssid():
    """
    Gets the SSID of the currently active Wi-Fi connection using nmcli.
    Returns the SSID as a string, or None if not connected or an error occurs.
    """
    try:
        # This command uses nmcli's terse output mode (-t) to find active Wi-Fi
        # connections and extracts the SSID.
        # `nmcli -t -f active,ssid dev wifi` -> lists all visible wifi networks, e.g., "yes:MySSID" or "no:AnotherSSID"
        # `egrep '^yes'` -> filters for the line starting with "yes"
        # `cut -d':' -f2` -> splits the line by ":" and takes the second part (the SSID)
        cmd = "nmcli -t -f active,ssid dev wifi | egrep '^yes' | cut -d':' -f2"
        result = subprocess.run(
            cmd,
            shell=True,  # shell=True is needed for the pipe `|`
            capture_output=True,
            text=True,
            check=False # Don't throw error if no wifi is active
        )
        if result.returncode != 0:
            # This is common if not connected to any Wi-Fi.
            print("Info: Could not determine current Wi-Fi SSID (nmcli command failed). Possibly not connected.")
            return None

        ssid = result.stdout.strip()
        return ssid if ssid else None

    except FileNotFoundError:
        print("ERROR: 'nmcli' or other shell commands not found. Is NetworkManager installed?")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting current SSID: {e}")
        return None

def configure_wifi(ssid, password):
    """
    Configures the system to connect to a new Wi-Fi network using nmcli.
    This requires sudo privileges.
    """
    print(f"Attempting to configure and connect to Wi-Fi network: '{ssid}'")
    try:
        # The `nmcli dev wifi connect` command is robust. It will:
        # 1. Look for an existing connection profile for the SSID.
        # 2. If found, it will use it to connect.
        # 3. If not found, it will create a new profile with the given password and connect.
        command = ['sudo', '-E', 'nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True # Throw an exception if nmcli returns a non-zero exit code
        )
        print("nmcli command executed successfully.")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to configure Wi-Fi using nmcli.")
        print(f"  Return code: {e.returncode}")
        # nmcli often provides helpful error messages in stderr.
        print(f"  Stderr: {e.stderr.strip()}")
        print("  This could be due to an incorrect password, the network being out of range, or permissions issues.")
        return False
    except FileNotFoundError:
        print("ERROR: 'nmcli' or 'sudo' command not found. Is NetworkManager installed and are you on a Linux system?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Wi-Fi configuration: {e}")
        return False

def check_and_configure_wifi():
    """
    Main logic: checks for netconfig.json on the SD card and applies the
    configuration if it's different from the current network.
    """
    print("\n--- Checking for network configuration file... ---")
    if not os.path.exists(NETCONFIG_FILE_PATH):
        print(f"Info: Network configuration file not found at '{NETCONFIG_FILE_PATH}'. Skipping Wi-Fi setup.")
        return

    print(f"Found network configuration file: '{NETCONFIG_FILE_PATH}'")
    try:
        with open(NETCONFIG_FILE_PATH, 'r') as f:
            net_config = json.load(f)
        
        new_ssid = net_config.get("ssid")
        new_pwd = net_config.get("pwd")

        if not new_ssid or not new_pwd:
            print("ERROR: 'ssid' or 'pwd' missing from netconfig.json. Cannot configure network.")
            return

    except (json.JSONDecodeError, Exception) as e:
        print(f"ERROR: Failed to read or parse '{NETCONFIG_FILE_PATH}': {e}")
        return

    current_ssid = get_current_wifi_ssid()
    print(f"Current Wi-Fi SSID: '{current_ssid}'")
    print(f"Target Wi-Fi SSID:  '{new_ssid}'")

    if current_ssid == new_ssid:
        print(f"Already connected to the target Wi-Fi network ('{new_ssid}'). No action needed.")
        return

    if configure_wifi(new_ssid, new_pwd):
        print("Waiting 5 seconds for connection to establish...")
        time.sleep(5)
        final_ssid = get_current_wifi_ssid()
        if final_ssid == new_ssid:
            print(f"--- SUCCESS: Now connected to '{final_ssid}'. ---")
        else:
            print(f"--- WARNING: Failed to connect to '{new_ssid}'. Current connection is '{final_ssid}'. Please check credentials and network availability. ---")

if __name__ == "__main__":
    print("--- Standalone Network Configuration Utility ---")
    print("This script attempts to read Wi-Fi credentials from")
    print(f"'{NETCONFIG_FILE_PATH}' and configure the system's network.")
    print("\nIMPORTANT:")
    print("1. This script requires 'sudo' privileges to run 'nmcli'.")
    print("   You may be prompted for your password.")
    print("2. The SD card MUST be mounted at /mnt/sdcard before running this.")
    print("   Example: sudo mount /dev/mmcblk1p1 /mnt/sdcard")
    
    check_and_configure_wifi()
