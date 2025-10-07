import os
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel # Import GenerativeModel
 
# Path to your service account key file
KEY_PATH = '/Users/sydneybach/sydney-bach.json'
# KEY_PATH = 'bach-sa.json'
 
# Project details - replace with your actual details
PROJECT_ID = 'massmkt-poc'
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-pro'
 
# Authenticate using the service account key
credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
 
# Initialize Vertex AI with the credentials
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
 
# Get the model
model = GenerativeModel(model_name=MODEL_NAME)  # Use GenerativeModel directly from the import
 
# Example text prompt
prompt = "Summarize the plot of 'I am Legend'."
 
# Generate content
response = model.generate_content(prompt)
 
# Print the response
print(response.text)