# test_key.py
import os
import google.generativeai as genai

try:
    # The script will automatically look for the GOOGLE_API_KEY variable
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("API Key found and configured successfully!")
    
    print("\nAttempting to connect to Google AI and list available models...")
    
    # This is the actual test: asking Google for a list of models
    for m in genai.list_models():
        # We only care about models that can be used for embeddings
        if 'embedContent' in m.supported_generation_methods:
            print(f"- Found a working model: {m.name}")
    
    print("\nSUCCESS: Your API key is correct and working!")

except Exception as e:
    # If any part of this fails, it will print an error
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Failed to connect or authenticate with Google AI.")
    print(f"Error details: {e}")
    print("\nPlease check that your API key is correct and has no typos.")