import pandas as pd
from openai import OpenAI
import json
import os
import time
import argparse

# Initialize OpenAI client
client = OpenAI(
    api_key="..."
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

def load_assistant_id() -> str:
    """Load the assistant ID from the file if it exists."""
    if os.path.exists(ASSISTANT_ID_FILE):
        with open(ASSISTANT_ID_FILE, 'r') as file:
            return file.read().strip()
    return None

def create_or_get_assistant(guidelines: str) -> str:
    """Check for an existing assistant ID or create a new assistant if none exists."""
    assistant_id = load_assistant_id()
    
    if assistant_id:
        print(f"Reusing existing assistant ID: {assistant_id}")
        return assistant_id

    print("Creating a new assistant...")
    instructions = f"""
    You are an assistant that annotates emotions in app reviews. Follow these instructions strictly:

    - Always return a JSON object following this schema:
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

    - Do not add any text, explanations, or formatting outside the JSON.
    - If uncertain, set the value to 0.
    - Always return the JSON as a single valid object.

    {guidelines}
    """

    assistant = client.beta.assistants.create(
        name="Emotion Annotation Assistant",
        instructions=instructions,
        model="gpt-4o",
        tools=[]
    )

    save_assistant_id(assistant.id)
    return assistant.id

def create_thread() -> str:
    """Create a new thread for conversation persistence."""
    thread = client.beta.threads.create()
    return thread.id

def validate_json(response_text: str) -> dict:
    """Validate and clean the JSON response to match the expected schema."""
    expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                         'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    try:
        response_json = json.loads(response_text)
        
        # Ensure all expected emotions exist in the output
        for emotion in expected_emotions:
            if emotion not in response_json or not isinstance(response_json[emotion], int):
                response_json[emotion] = 0  # Default missing or incorrect values to 0
        
        return response_json

    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON. Returning default values.")
        return {emotion: 0 for emotion in expected_emotions}

def get_gpt4o_annotation(thread_id: str, assistant_id: str, review: str, sentence: str) -> dict:
    """Use the Assistants API to classify emotions with persistent context."""
    
    user_prompt = json.dumps({
        "review": review,
        "sentence": sentence
    }, ensure_ascii=False)

    try:
        # Add message to thread
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_prompt
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        # Wait for completion
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            time.sleep(2)  # Avoid excessive API calls

        # Retrieve response
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        last_message = messages.data[0].content[0].text.value

        # Validate and parse JSON response
        return validate_json(last_message)

    except Exception as e:
        print(f"Error processing review: {e}")
        return {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}

def main(input_file: str, guidelines_file: str, output_folder: str, n: int = None):
    """
    Process annotations using the Assistants API.

    Args:
        input_file (str): Path to the input Excel file
        guidelines_file (str): Path to the guidelines text file
        output_folder (str): Path to the output directory
        n (int, optional): Number of rows to process. If None, process all rows.
    """

    # Read the Excel file and load guidelines
    df = pd.read_excel(input_file)
    
    if n is not None:
        df = df.head(n)  # Limit rows if n is specified
    
    guidelines = load_guidelines(guidelines_file)

    # Get existing assistant or create a new one
    assistant_id = create_or_get_assistant(guidelines)

    # Create a thread for persistent context
    thread_id = create_thread()
    
    # Emotion columns
    emotion_columns = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                       'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']

    # Process each review
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")

        annotations = get_gpt4o_annotation(
            thread_id=thread_id,
            assistant_id=assistant_id,
            review=row['review'],
            sentence=row['sentence']
        )

        # Update DataFrame with annotations
        for emotion in emotion_columns:
            df.at[idx, emotion] = annotations.get(emotion, 0)

        # Respect API rate limits
        time.sleep(2)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save results
    output_file = os.path.join(output_folder, 'GPT-4o-annotations.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Annotation completed! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-4o emotion annotation with Assistants API')
    parser.add_argument('--input', required=True, help='Path to input Excel file')
    parser.add_argument('--guidelines', required=True, help='Path to guidelines text file')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    
    args = parser.parse_args()
    main(args.input, args.guidelines, args.output, args.n)
