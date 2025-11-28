#!/usr/bin/env python3
import cv2
import argparse
import sys

def test_camera(device_index):
    """
    Attempts to open a camera at a specific index, grab one frame,
    and report success or failure.
    """
    print(f"\n--- Testing camera at index: {device_index} ---")
    
    # The second argument (backend) was removed for compatibility with older
    # versions of OpenCV, which only accept the device index.
    cap = cv2.VideoCapture(device_index)
    
    if not cap.isOpened():
        print(f"RESULT: FAILURE. Could not open camera at index {device_index}.")
        print("        This could be due to permissions, the camera being in use by another process, or an incorrect index.")
        cap.release()
        return

    print("INFO: Camera opened successfully. Attempting to grab a frame...")
    
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("RESULT: FAILURE. Camera opened, but failed to grab a frame.")
        print("        This often indicates a driver issue, a problem with the camera's data stream, or a hardware fault.")
    else:
        height, width, _ = frame.shape
        print(f"RESULT: SUCCESS! Frame grabbed successfully from camera index {device_index}.")
        print(f"        Resolution: {width}x{height}")
    
    print("INFO: Releasing camera.")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to test a camera with OpenCV.")
    parser.add_argument('--device', type=int, required=True, help="The device index of the camera to test (e.g., 0, 1, 2).")
    args = parser.parse_args()
    
    test_camera(args.device)