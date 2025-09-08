import pyttsx3

def list_voices():
    """
    Initializes the TTS engine and prints details for all available voices.
    """
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print("--- Available TTS Voices ---")
        for i, voice in enumerate(voices):
            print(f"\n--- Voice Index: {i} ---")
            print(f"  ID: {voice.id}")
            print(f"  Name: {voice.name}")
            print(f"  Languages: {voice.languages}")
            print(f"  Gender: {voice.gender}")
        engine.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure 'espeak' is installed (`sudo apt-get install espeak`).")

if __name__ == "__main__":
    list_voices()
