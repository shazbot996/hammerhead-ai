#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import argparse

def check_audio_devices():
    """
    Lists all available audio devices. If a device index is specified,
    it also performs a quick recording test on that device.
    """
    parser = argparse.ArgumentParser(
        description="Audio device diagnostic tool.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        '--device', type=int,
        help="The index of the audio input device to test.\nIf not provided, the script will only list available devices."
    )
    args = parser.parse_args()

    print("--- Audio Device Check ---")
    try:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
    except Exception as e:
        print(f"\nERROR: Could not query audio devices. {e}")
        print("This may indicate a problem with your system's audio drivers (PortAudio).")
        print("Please ensure audio libraries are installed correctly.")
        return

    # If no device is specified, just list them and exit with helpful instructions.
    if args.device is None:
        print("\n--- Usage ---")
        print("To test a specific microphone, re-run this script with the --device flag.")
        print("Example: python3 check_audio.py --device 1")
        print("\nLook for a device with a non-zero number of 'in' channels in the list above.")
        print("Once you find a working device index, add it to your config.json under 'hotword_config'.")
        print('Example config.json entry: "device_index": 1')
        return

    device_to_test = args.device

    try:
        # --- Perform a short recording test ---
        sample_rate = 16000  # Standard sample rate for speech
        duration = 3  # seconds

        # Check if the selected device has input channels
        device_info = sd.query_devices(device_to_test)
        if device_info.get('max_input_channels', 0) == 0:
            print(f"\nERROR: Device index {device_to_test} ('{device_info['name']}') has no input channels and cannot be used for recording.")
            return

        print(f"\n--- Starting a {duration}-second recording test on device index {device_to_test} ('{device_info['name']}')... ---")
        print("Please speak into your microphone now.")

        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_to_test)
        sd.wait()  # Wait for recording to complete

        print("Recording complete.")

        # --- Analyze the recording ---
        max_amplitude = np.max(np.abs(recording))

        print(f"\n--- Analysis ---")
        print(f"Max audio level (amplitude) recorded: {max_amplitude}")

        # A reasonable threshold for speech, can be adjusted.
        # A silent room might be ~100-300, speech should be > 500.
        if max_amplitude > 500:
            print(f"\nSUCCESS: Audio was detected on device {device_to_test}! Your microphone appears to be working correctly.")
            print(f"You should set '\"device_index\": {device_to_test}' in your config.json.")
        else:
            print(f"\nWARNING: No significant audio was detected on device {device_to_test}.")
            print("The microphone might be muted, the gain set too low, or it may not be functioning.")
            print("Try speaking louder or checking your system's audio settings.")

    except Exception as e:
        print(f"\nAn error occurred during the recording test on device {device_to_test}: {e}")
        print("Please ensure the device index is correct and the microphone is properly connected.")

if __name__ == "__main__":
    check_audio_devices()