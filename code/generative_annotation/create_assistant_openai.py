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
    You are an assistant that annotates emotions in app reviews. You have been provided detailed guidelines below that you MUST follow at all times.

    **Annotation Guidelines:**
    {guidelines}

    **Annotation Output Format:**
    Return a JSON object following this schema:
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

    **Important Rules:**
    - Follow the annotation guidelines strictly.
    - Do NOT include explanations, formatting, or additional text.
    - Ensure the JSON is always syntactically valid.
    """

    # Create Assistant
    assistant = client.beta.assistants.create(
        name="Emotion Annotation Assistant",
        instructions=instructions,
        model=model,
        temperature=0.0,
        response_format={ "type": "json_object" }

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
