import pandas as pd
from openai import OpenAI
import json
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASSISTANT_ID_FILE = "assistant_id.txt"

def load_assistant_id() -> str:
    """Load the assistant ID from a file if it exists."""
    if os.path.exists(ASSISTANT_ID_FILE):
        with open(ASSISTANT_ID_FILE, 'r') as file:
            return file.read().strip()
    return None

def get_assistant() -> str:
    """Check for an existing assistant ID or raise an error if not found."""
    assistant_id = load_assistant_id()
    
    if assistant_id:
        print(f"Reusing existing assistant ID: {assistant_id}")
        return assistant_id
    
    raise ValueError("Assistant ID not found. Run create_assistant.py first.")

def create_thread() -> str:
    """Create a new thread for conversation persistence."""
    return client.beta.threads.create().id

def validate_json(response_text: str) -> dict:
    """Ensure the response is a valid JSON object and matches the expected schema."""
    expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                         'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]

        response_json = json.loads(cleaned_text.strip())

        for emotion in expected_emotions:
            if emotion not in response_json or not isinstance(response_json[emotion], int):
                response_json[emotion] = 0  

        return response_json

    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON. Returning default values.")
        print(response_text)
        return {emotion: 0 for emotion in expected_emotions}

def get_annotation(assistant_id: str, review: str, sentence: str) -> dict:
    """Classify emotions without using RAG, minimizing token usage."""
    user_prompt = json.dumps({"review": review, "sentence": sentence}, ensure_ascii=False)
    thread_id = create_thread()

    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                print(run_status)
            time.sleep(2)

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        last_message = messages.data[0].content[0].text.value

        annotations = validate_json(last_message)

        prompt_tokens = getattr(run_status.usage, "prompt_tokens", 0)
        completion_tokens = getattr(run_status.usage, "completion_tokens", 0)
        total_tokens = getattr(run_status.usage, "total_tokens", 0)
        annotations["prompt_tokens"] = prompt_tokens
        annotations["completion_tokens"] = completion_tokens
        annotations["total_tokens"] = total_tokens  

        return annotations

    except Exception as e:
        print(f"Error processing review: {e}")
        return {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}

def main(input_file: str, output_folder: str, n: int = None):
    print(f"Starting annotation process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_excel(input_file)
    if n is not None:
        df = df.head(n)

    assistant_id = get_assistant()

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'GPT-4o-annotations.xlsx')

    results_df = pd.DataFrame()
    total_tokens_used = 0  
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0

    for idx, row in df.iterrows():
        review_start_time = time.time()
        print(f"Processing review {idx + 1}/{len(df)}")

        annotations = get_annotation(assistant_id, row['review'], row['sentence'])

        annotation_time = time.time() - review_start_time
        annotations["annotation_time"] = annotation_time

        
        total_tokens_used += annotations.get("total_tokens", 0)
        total_prompt_tokens_used += annotations.get("prompt_tokens", 0)
        total_completion_tokens_used += annotations.get("completion_tokens", 0)

        row.update(annotations)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Review {idx + 1} processed (Time: {annotation_time:.2f}s, Total Tokens: {annotations.get('total_tokens', 0)}, Prompt Tokens: {annotations.get('prompt_tokens', 0)}, Completion Tokens: {annotations.get('completion_tokens', 0)})")
        time.sleep(12)

    print(f"Total tokens used: {total_tokens_used}")
    print(f"Total prompt tokens used: {total_prompt_tokens_used}")
    print(f"Total completion tokens used: {total_completion_tokens_used}")
    print("Annotation process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emotion annotation with Assistants API')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    args = parser.parse_args()
    main(args.input, args.output, args.n)
