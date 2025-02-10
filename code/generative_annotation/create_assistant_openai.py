from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

ASSISTANT_ID_FILE = "assistant_id.txt"

def load_guidelines(guidelines_path: str) -> str:
    """Load annotation guidelines from a text file."""
    with open(guidelines_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_assistant_id(assistant_id: str):
    """Save the assistant ID to a file for reuse."""
    with open(ASSISTANT_ID_FILE, 'w') as file:
        file.write(assistant_id)

def create_assistant(guidelines_file: str, model: str = "gpt-4o"):
    """Create an OpenAI assistant with guidelines stored in memory."""
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
    assistant = client.beta.assistants.create(
        name="Emotion Annotation Assistant",
        instructions=instructions,
        model=model,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classify_review_emotions",
                "description": "Classifies a list of app review sentences into emotion categories, returning a structured JSON array.",
                "schema": {
                  "type": "object",
                  "properties": {
                    "reviews": {  # Explicitly add the "schema" key
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "Joy": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Joy, 0 otherwise." },
                              "Trust": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Trust, 0 otherwise." },
                              "Fear": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Fear, 0 otherwise." },
                              "Surprise": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Surprise, 0 otherwise." },
                              "Sadness": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Sadness, 0 otherwise." },
                              "Disgust": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Disgust, 0 otherwise." },
                              "Anger": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Anger, 0 otherwise." },
                              "Anticipation": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Anticipation, 0 otherwise." },
                              "Neutral": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence expresses Neutral sentiment, 0 otherwise." },
                              "Reject": { "type": "integer", "enum": [0, 1], "description": "1 if the sentence is rejected, 0 otherwise." }
                          },
                          "required": ["sentence", "Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral", "Reject"]
                      }
                    }
                  }, 
                  "required": [
                    "reviews"
                  ]
                }
            }
        }
    )

    save_assistant_id(assistant.id)

    print("Assistant created successfully.")
    return assistant

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an OpenAI assistant for emotion annotation.')
    parser.add_argument('--guidelines', required=True, help='Path to the guidelines file')
    parser.add_argument('--model', default="gpt-4o", help='OpenAI model to use')

    args = parser.parse_args()
    create_assistant(args.guidelines, args.model)
