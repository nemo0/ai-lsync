# import requests
# import os

# # --- Configuration ---
# # Replace this with the URL provided by `modal deploy`
# MODAL_URL = "https://nemo0--latentsync-api-web-server.modal.run/lipsync" 

# # Define the paths to your local video and audio files
# # Make sure these files exist in the same directory as this script.
# # You can use the demo files from the LatentSync repository.
# VIDEO_FILE_PATH = "assets/demo1_video.mp4"
# AUDIO_FILE_PATH = "assets/demo1_audio.wav"
# OUTPUT_VIDEO_PATH = "output.mp4"

# def test_lipsync_api():
#     """
#     Sends a video and audio file to the deployed LatentSync API
#     and saves the resulting video.
#     """
#     print(f"Sending request to: {MODAL_URL}")
#     print(f"Video file: {VIDEO_FILE_PATH}")
#     print(f"Audio file: {AUDIO_FILE_PATH}")

#     # Check if the input files exist
#     if not os.path.exists(VIDEO_FILE_PATH):
#         print(f"Error: Video file not found at '{VIDEO_FILE_PATH}'")
#         print("Please download the demo assets from the LatentSync repo or use your own files.")
#         return
#     if not os.path.exists(AUDIO_FILE_PATH):
#         print(f"Error: Audio file not found at '{AUDIO_FILE_PATH}'")
#         print("Please download the demo assets from the LatentSync repo or use your own files.")
#         return

#     try:
#         # Prepare the files for the multipart/form-data request
#         files = {
#             'video': (os.path.basename(VIDEO_FILE_PATH), open(VIDEO_FILE_PATH, 'rb'), 'video/mp4'),
#             'audio': (os.path.basename(AUDIO_FILE_PATH), open(AUDIO_FILE_PATH, 'rb'), 'audio/wav')
#         }

#         # Send the POST request to the API
#         response = requests.post(MODAL_URL, files=files, timeout=600) # 10-minute timeout

#         # Check the response
#         if response.status_code == 200:
#             # Save the returned video content to a file
#             with open(OUTPUT_VIDEO_PATH, 'wb') as f:
#                 f.write(response.content)
#             print(f"Success! Lip-synced video saved to '{OUTPUT_VIDEO_PATH}'")
#         else:
#             # Print an error message if the request failed
#             print(f"Error: API returned status code {response.status_code}")
#             print("Response:")
#             print(response.text)

#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred while sending the request: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     if "your-modal-app-url" in MODAL_URL:
#         print("Please update the 'MODAL_URL' variable in this script with your deployed app's URL.")
#     else:
#         test_lipsync_api()


