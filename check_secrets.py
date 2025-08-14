# check_secrets.py (A dedicated tool to diagnose the secrets.toml issue)

import os
import sys

# We use the 'toml' library, which is what Streamlit uses under the hood.
# This ensures we are testing the exact same way Streamlit does.
try:
    import toml
except ImportError:
    print("\nFATAL ERROR: The 'toml' library is not installed.")
    print("Please run: pip install toml")
    sys.exit(1)

print("--- Starting Secrets.toml Diagnostic Check ---")

# --- Test 1: Check Current Location ---
current_directory = os.getcwd()
print(f"\n[INFO] Running diagnostic from directory: {current_directory}")

# --- Test 2: Check for .streamlit folder ---
streamlit_folder_path = os.path.join(current_directory, ".streamlit")
print(f"[INFO] Checking for folder at: {streamlit_folder_path}")

if os.path.isdir(streamlit_folder_path):
    print("✅ SUCCESS: Found the '.streamlit' folder.")
else:
    print("❌ FAILURE: The '.streamlit' folder was NOT FOUND in this directory.")
    print("SOLUTION: Please create the '.streamlit' folder in your main project directory.")
    sys.exit(1)

# --- Test 3: Check for secrets.toml file ---
secrets_file_path = os.path.join(streamlit_folder_path, "secrets.toml")
print(f"[INFO] Checking for secrets file at: {secrets_file_path}")

if os.path.isfile(secrets_file_path):
    print("✅ SUCCESS: Found the 'secrets.toml' file.")
else:
    print("❌ FAILURE: The 'secrets.toml' file was NOT FOUND inside the '.streamlit' folder.")
    print("SOLUTION: Please create the 'secrets.toml' file inside your '.streamlit' folder.")
    sys.exit(1)

# --- Test 4: Try to Read and Parse the file ---
print("[INFO] Attempting to read and parse the 'secrets.toml' file...")
try:
    with open(secrets_file_path, "r") as f:
        secrets_content = toml.load(f)
    print("✅ SUCCESS: The 'secrets.toml' file is a valid TOML file.")
except Exception as e:
    print(f"❌ FAILURE: The 'secrets.toml' file has a syntax error and could not be read.")
    print(f"   ERROR DETAILS: {e}")
    print("SOLUTION: Please open the file and ensure it has no typos and follows the correct format.")
    sys.exit(1)

# --- Test 5: Check for the required [api_keys] section ---
print("[INFO] Checking for the '[api_keys]' section inside the file...")
if "api_keys" in secrets_content and isinstance(secrets_content["api_keys"], dict):
    print("✅ SUCCESS: Found the '[api_keys]' section.")
else:
    print("❌ FAILURE: The '[api_keys]' section is MISSING or NOT FORMATTED correctly.")
    print("SOLUTION: Your secrets.toml file MUST start with the line '[api_keys]'.")
    sys.exit(1)
    
# --- Test 6: Check for the 'google' and 'openai' keys ---
api_keys_section = secrets_content["api_keys"]
print("[INFO] Checking for the 'google' and 'openai' keys under '[api_keys]'...")

google_key = api_keys_section.get("google")
openai_key = api_keys_section.get("openai")

if google_key and isinstance(google_key, str) and len(google_key) > 10:
    print("✅ SUCCESS: Found 'google' key.")
else:
    print("❌ FAILURE: The 'google' key is MISSING or EMPTY under the '[api_keys]' section.")
    print("SOLUTION: Ensure your file has a line like: google = \"AIzaSy...\"")

if openai_key and isinstance(openai_key, str) and len(openai_key) > 10:
    print("✅ SUCCESS: Found 'openai' key.")
else:
    print("❌ FAILURE: The 'openai' key is MISSING or EMPTY under the '[api_keys]' section.")
    print("SOLUTION: Ensure your file has a line like: openai = \"sk-...\"")

print("\n--- Diagnostic Complete ---")
