import os
import json
import time
import re
import cv2
import argparse
import threading
import pickle
import face_recognition
import numpy as np
import pyttsx3
import subprocess
try:
    from google.cloud import texttospeech
except ImportError:
    print("Warning: Google Cloud TextToSpeech library not found. Google TTS will not be available.")

# --- Configuration ---
CONFIG_FILE_NAME = "config.json"

LOCAL_CACHE_DIR = "local_data_cache"
LOCAL_FACES_DIR = os.path.join(LOCAL_CACHE_DIR, "face-index")
LOCAL_CONFIG_DIR = os.path.join(LOCAL_CACHE_DIR, "config")

# Cooldown in seconds to prevent spamming responses for the same person
RESPONSE_COOLDOWN = 60  # 1 minute
TEMP_AUDIO_FILE_PREFIX = "temp_response" # Used for thread-safe filenames

# --- 1. Load Local Data ---
def load_data():
    """
    Loads configuration from a local JSON file.
    Assumes face images are already present in the local faces directory.
    """
    print("Loading local configuration...")

    # Create local directories
    os.makedirs(LOCAL_FACES_DIR, exist_ok=True)
    os.makedirs(LOCAL_CONFIG_DIR, exist_ok=True)

    config_data = {}
    local_config_path = os.path.join(LOCAL_CONFIG_DIR, CONFIG_FILE_NAME)

    if os.path.exists(local_config_path):
        with open(local_config_path, 'r') as f:
            config_data = json.load(f)
        print("Configuration loaded successfully.")
    else:
        print(f"ERROR: Config file not found at '{local_config_path}'")
        print(f"Please create a '{CONFIG_FILE_NAME}' file in the '{LOCAL_CONFIG_DIR}' directory.")

    return config_data

# --- 2. Index Faces ---
def index_faces():
    """
    Loads face encodings from a cache file if it's valid, otherwise creates
    them from images and saves them to the cache. The cache is considered
    invalid if the set of image filenames has changed.
    """
    cache_path = os.path.join(LOCAL_CACHE_DIR, "face_encodings.pkl")

    # Get the current list of image files to validate against the cache
    try:
        current_image_files = sorted([
            f for f in os.listdir(LOCAL_FACES_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"ERROR: Local faces directory not found: {LOCAL_FACES_DIR}")
        return [], []

    # --- 1. Try to load from cache and validate it ---
    if os.path.exists(cache_path):
        print("Loading face encodings from cache...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            # Validate cache by comparing file lists
            if set(cached_data.get('filenames', [])) == set(current_image_files):
                print("Cache is valid. Loading encodings from cache.")
                known_face_encodings = cached_data['encodings']
                known_face_names = cached_data['names']
                print(f"Cache loaded. Found {len(known_face_encodings)} indexed faces.")
                return known_face_encodings, known_face_names
            else:
                print("Cache is stale (image files have changed). Re-indexing...")
        except Exception as e:
            print(f"Warning: Could not load or validate cache file ({e}). Re-indexing from images.")

    # --- 2. If cache fails or doesn't exist, index from images ---
    print("Indexing faces from image files...")
    known_face_encodings = []
    known_face_names = []
    processed_filenames = []

    for filename in current_image_files:
        path = os.path.join(LOCAL_FACES_DIR, filename)
        name = re.split(r'[-\d._]', os.path.splitext(filename)[0])[0]
        if not name:
            print(f"Could not parse name from filename: {filename}. Skipping.")
            continue
        print(f"Processing {filename} for person: {name}")
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            processed_filenames.append(filename)
        else:
            print(f"Warning: No face found in {filename}. Skipping.")
    # --- 3. Save the newly generated encodings and file list to the cache ---
    if known_face_encodings:
        print(f"Face indexing complete. Found {len(known_face_encodings)} face(s).")
        print(f"Saving new encodings to cache file: {cache_path}")
        cache_data = {
            'filenames': processed_filenames,
            'encodings': known_face_encodings,
            'names': known_face_names
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    else:
        print("Face indexing complete. No faces were found to cache.")

    return known_face_encodings, known_face_names

# --- 3. Play Response ---
def _play_audio_threaded(response_text, config_data, tts_client):
    """
    This function runs in a separate thread to play audio without blocking the main loop.
    It contains the actual blocking TTS logic.
    """
    print(f"DEBUG (Thread): Attempting to use '{config_data.get('tts_engine', 'pyttsx3')}' engine.")
    tts_engine_choice = config_data.get("tts_engine", "pyttsx3")
    temp_file = f"{TEMP_AUDIO_FILE_PREFIX}-{threading.get_ident()}.wav" # Use WAV format for better compatibility
    try:
        if tts_engine_choice == "google_cloud" and tts_client:
            # --- Google Cloud TTS Logic ---
            tts_config = config_data.get("tts_config", {}).get("google_cloud", {})
            synthesis_input = texttospeech.SynthesisInput(text=response_text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=tts_config.get("language_code", "en-US"),
                name=tts_config.get("voice_name", "en-US-Wavenet-D")
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16, # Request WAV format
                speaking_rate=tts_config.get("speaking_rate", 1.0),
                sample_rate_hertz=tts_config.get("sample_rate_hertz", 22050), # Specify sample rate for WAV
                pitch=tts_config.get("pitch", 0.0)
            )
            print(f"DEBUG (Thread): Synthesizing speech with Google Cloud using config: {tts_config}")
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            # 1. Write the entire audio content to a file. The 'with' statement ensures it's fully written and closed.
            with open(temp_file, "wb") as out:
                out.write(response.audio_content)
            
            # --- Diagnostic Check: Ensure the audio file is not empty ---
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                print("ERROR (Thread): Synthesized audio file is empty or was not created.")
                return

            # --- Play the WAV file using aplay, the standard Linux audio player ---
            try:
                print(f"DEBUG (Thread): Playing {temp_file} with aplay...")
                subprocess.run(
                    ['aplay', '-q', temp_file], # The '-q' flag makes it quiet
                    check=True,
                    stderr=subprocess.PIPE # Capture errors if any
                )
            except FileNotFoundError:
                print("\nERROR: 'aplay' command not found. This is highly unusual for a Linux system.\n")
            except subprocess.CalledProcessError as e:
                # This will catch errors if aplay returns a non-zero exit code
                print(f"ERROR (Thread): aplay failed with exit code {e.returncode}.")
                print(f"  mpg123 stderr: {e.stderr}")

        elif tts_engine_choice == "pyttsx3" and tts_client:
            # --- pyttsx3 (local) Logic ---
            # NOTE: pyttsx3 is not strictly thread-safe. While this works in many
            # cases, a more complex implementation might be needed if issues arise.
            print("DEBUG (Thread): Synthesizing speech with pyttsx3.")
            tts_client.say(response_text)
            tts_client.runAndWait()
        else:
            print(f"ERROR (Thread): TTS engine '{tts_engine_choice}' not supported or client not initialized.")

    except Exception as e:
        print(f"ERROR (Thread): Failed to play TTS response: {e}")
    finally:
        # 3. Clean up the temp file after playback is guaranteed to be finished.
        if os.path.exists(temp_file):
            os.remove(temp_file)

def play_response(name, config_data, tts_client, last_played_times):
    """
    Looks up a response and starts a new thread to play it using text-to-speech.
    Includes a cooldown to prevent spam.
    """
    current_time = time.time()

    # Check cooldown
    if name in last_played_times and (current_time - last_played_times[name]) < RESPONSE_COOLDOWN:
        return

    # Look for responses within the 'responses' key of the config data
    responses = config_data.get("responses", {})
    response_text = responses.get(name, responses.get("unknown"))

    if not response_text:
        return

    print(f"Recognized {name}. Playing response: '{response_text}'")

    # Update cooldown time immediately to prevent re-triggering while audio plays
    last_played_times[name] = current_time

    # Create and start a new thread for audio playback to avoid blocking the video loop
    audio_thread = threading.Thread(
        target=_play_audio_threaded,
        args=(response_text, config_data, tts_client)
    )
    audio_thread.daemon = True  # Allows main program to exit even if thread is running
    audio_thread.start()
# --- 4. Process Frame ---
def process_frame(frame, known_face_encodings, known_face_names):
    """
    Detects and recognizes faces in a single video frame.

    NOTE: This function uses face_recognition's built-in HOG-based detector.
    For better performance on a Coral Dev Board, this should be replaced with
    a PyCoral-accelerated face *detection* model. The detected face bounding
    boxes would then be passed to face_recognition for *encoding* and comparison.
    """
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    recognized_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        recognized_names.append(name)

    # --- For visualization (can be removed for performance) ---
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return recognized_names, frame


# --- 5. Helper Functions ---
def find_camera_index():
    """
    Iterates through video device indices to find the first available camera.
    This is more robust than hardcoding index 0.
    """
    print("Searching for available camera...")
    # Check indices 0 through 9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}.")
            cap.release()
            return i
        cap.release()
    return -1 # Return -1 if no camera is found


# --- 6. Main Orchestrator ---
def main():
    """
    Main function to set up and run the facial recognition application.
    """
    # --- Argument Parsing for Test Mode ---
    parser = argparse.ArgumentParser(description="Facial Recognition with TTS response.")
    parser.add_argument('--test_tts', type=str, help="Run a TTS test with the given phrase and exit.")
    args = parser.parse_args()

    # --- Initialization ---
    try:
        config_data = load_data()
        if not config_data:
            return  # Exit if config loading failed

        known_face_encodings, known_face_names = index_faces()
        if not known_face_encodings:
            print(f"No faces were indexed. Please check the '{LOCAL_FACES_DIR}' directory.")
            return

        # --- Initialize the configured TTS engine ---
        tts_engine_choice = config_data.get("tts_engine", "pyttsx3")
        tts_client = None

        # If using Google Cloud, set the credentials environment variable
        if tts_engine_choice == "google_cloud":
            key_path = config_data.get("gcp_service_account_key_path")
            print(f"  - Checking for GCP key at: {key_path}")
            if key_path and os.path.exists(key_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
                print("  - GCP service account credentials SET from config file.")
            else:
                print("  - WARNING: GCP service account key NOT FOUND at the specified path.")


        if tts_engine_choice == "google_cloud":
            print("Initializing Google Cloud Text-to-Speech client...")
            try:
                tts_client = texttospeech.TextToSpeechClient()
                print("Google Cloud TTS client initialized successfully.")
            except Exception as e:
                print(f"ERROR: Failed to initialize Google Cloud TTS. Check credentials/internet. {e}")
                print("Falling back to local pyttsx3 engine.")
                tts_engine_choice = "pyttsx3"

        if tts_engine_choice == "pyttsx3":
            print("Initializing local pyttsx3 Text-to-Speech engine...")
            tts_client = pyttsx3.init()
            # Apply custom pyttsx3 configuration if it exists
            pyttsx3_config = config_data.get("tts_config", {}).get("pyttsx3", {})
            if pyttsx3_config:
                print("Applying custom pyttsx3 configuration...")
                rate = pyttsx3_config.get("rate")
                if rate is not None:
                    tts_client.setProperty('rate', int(rate))
                volume = pyttsx3_config.get("volume")
                if volume is not None:
                    tts_client.setProperty('volume', float(volume))
                voice_id = pyttsx3_config.get("voice_id")
                if voice_id:
                    tts_client.setProperty('voice', voice_id)

        if tts_client is None:
            print("ERROR: Could not initialize any TTS engine. Audio will be disabled.")

        # --- Handle TTS Test Argument ---
        if args.test_tts:
            print(f"\n--- Running TTS Test ---")
            print(f"Input phrase: '{args.test_tts}'")
            if tts_client:
                # We can call the threaded function directly for the test
                _play_audio_threaded(args.test_tts, config_data, tts_client)
                print("--- TTS Test Complete ---")
            else:
                print("ERROR: Cannot run test, TTS client failed to initialize.")
            return # Exit after the test

        # --- Play Startup Greeting ---
        startup_greeting = config_data.get("startup_greeting")
        if startup_greeting and tts_client:
            print(f"\n--- Playing Startup Greeting ---")
            print(f"Message: '{startup_greeting}'")
            # We can use the same threaded player.
            # We don't need to join this thread, as it can play in the background
            # while the camera initializes.
            startup_audio_thread = threading.Thread(
                target=_play_audio_threaded,
                args=(startup_greeting, config_data, tts_client)
            )
            startup_audio_thread.daemon = True
            startup_audio_thread.start()

        # Initialize video capture
        print("Initializing camera...")
        camera_index = find_camera_index()
        if camera_index == -1:
            print("ERROR: No camera found. Please check hardware connection.")
            return

        video_capture = cv2.VideoCapture(camera_index)
        if not video_capture.isOpened():
            print("Error: Could not open video stream.")
            return

        # The software autofocus attempt below can cause an infinite loop of 'VIDIOC_QUERYCTRL'
        # errors on some drivers if the property is not supported. It is safer to
        # rely on manual lens focus and the camera's default startup behavior.
        # We will leave the warm-up period, as that is still beneficial.
        print("Skipping software autofocus attempt to prevent driver errors. Please use the manual focus ring on the lens.")

        # --- Camera warm-up period ---
        print("Letting camera warm up...")
        time.sleep(2.0)  # Wait 2 seconds for auto-exposure and focus to settle
        # Read and discard a few frames to clear the camera's buffer
        for _ in range(5):
            video_capture.read()

        print("\n--- Application starting ---")
        print("Press 'q' in the video window to quit.")

        # --- Set up the display window ---
        # For now, let's use a simple window to rule out any issues with fullscreen properties.
        cv2.namedWindow('Video')
        # Create a window that can be made full-screen
        # cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        # For a kiosk-like experience on the dev board, set it to full-screen
        # cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        return

    last_played_times = {}

    # --- Main Loop ---
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Process the frame for faces
        recognized_names, processed_frame = process_frame(frame, known_face_encodings, known_face_names)

        # Trigger audio response for each recognized person
        for name in recognized_names:
            play_response(name, config_data, tts_client, last_played_times)

        # Display the resulting image
        cv2.imshow('Video', processed_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Cleaning up and shutting down...")
    video_capture.release()
    cv2.destroyAllWindows()
    # Clean up any temp audio files that might have been orphaned if the app crashed
    for item in os.listdir('.'):
        if item.startswith(TEMP_AUDIO_FILE_PREFIX) and item.endswith(".wav"):
            os.remove(os.path.join('.', item))

if __name__ == "__main__":
    main()