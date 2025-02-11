import mistralai
import argparse
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Mistral client
client = mistralai.MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

def load_guidelines(guidelines_path: str) -> str:
    """Load annotation guidelines from a text file."""
    with open(guidelines_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_assistant_id(assistant_id: str, model: str):
    """Save the assistant ID to a file for reuse."""
    filename = f"assistant_id_{model}.txt"
    with open(filename, 'w') as file:
        file.write(assistant_id)

def create_assistant(guidelines_file: str, model: str = "mistral-large-latest"):
    """Create a Mistral AI assistant with guidelines stored in memory."""
    guidelines = load_guidelines(guidelines_file)

    print("Creating a new assistant...")

    instructions = f"""
    You are an assistant that annotates emotions in sentences from app reviews. You have been provided detailed guidelines below that you MUST follow at all times.

    **Annotation Guidelines:**
    {guidelines}

    **Annotation Input Format:**
    The input is a JSON list of objects following this schema:
    [
      {{
          "review": "The full text of the review",
          "sentence": "The sentence to annotate"
      }}
    ]

    **Annotation Output Format:**
    Return a JSON list of objects where each item corresponds to the input sentence in the same order, following this schema:
    [
      {{
          "Joy": 0 or 1,
          "Trust": 0 or 1,
          "Fear": 0 or 1,
          "Surprise": 0 or 1,
          "Sadness": 0 or 1,
          "Disgust": 0 or 1,
          "Anger": 0 or 1,
          "Anticipation": 0 or 1,
          "Neutral": 0 or 1,
          "Reject": 0 or 1
      }}
    ]

    **Important Rules:**
    - Each output object must directly correspond to an input sentence.
    - Ensure JSON validity and maintain correct key-value formatting.
    - Do NOT include explanations, additional text, or formatting.
    """

    # Create Assistant
    assistant = client.create_assistant(
        name="Emotion Annotation Assistant",
        instructions=instructions,
        model=model,
    )

    save_assistant_id(assistant['id'], model)
    print("Assistant created successfully.")
    return assistant

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a Mistral AI assistant for emotion annotation.')
    parser.add_argument('--guidelines', required=True, help='Path to the guidelines file')
    parser.add_argument('--model', default="mistral-large-latest", help='Mistral AI model to use')
    
    args = parser.parse_args()
    create_assistant(args.guidelines, args.model)
