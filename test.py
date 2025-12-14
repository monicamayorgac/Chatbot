import google.generativeai as genai
import os

# Paste your API key here just for this test
os.environ["GOOGLE_API_KEY"] = "AIzaSyCVGMgBvJ8UFkhnsPszSh-eLattoT8_GsM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("List of available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)