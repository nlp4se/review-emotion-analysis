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

def load_assistant_id(model: str) -> str:
    """Load the assistant ID from a model-specific file if it exists."""
    assistant_id_file = f"assistant_id_{model}.txt"
    if os.path.exists(assistant_id_file):
        with open(assistant_id_file, 'r') as file:
            return file.read().strip()
    return None

def get_assistant(model: str) -> str:
    """Check for an existing assistant ID or raise an error if not found."""
    assistant_id = load_assistant_id(model)
    
    if assistant_id:
        print(f"Reusing existing assistant ID for {model}: {assistant_id}")
        return assistant_id
    
    raise ValueError(f"Assistant ID not found for {model}. Run create_assistant.py first.")

def create_thread() -> str:
    """Create a new thread for conversation persistence."""
    return client.beta.threads.create().id

def get_annotation(assistant_id: str, reviews_batch: list) -> list:
    """Classify emotions for a batch of reviews."""
    user_prompt = json.dumps(reviews_batch, ensure_ascii=False)
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
            elif run_status.status != "in_progress" and run_status.status != "queued":
                print(run_status)
            time.sleep(2)

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        last_message = messages.data[0].content[0].text.value

        annotations_list = validate_json(last_message)
        
        prompt_tokens = getattr(run_status.usage, "prompt_tokens", 0)
        completion_tokens = getattr(run_status.usage, "completion_tokens", 0)
        total_tokens = getattr(run_status.usage, "total_tokens", 0)

        # Add token usage to each annotation in the batch
        for annotation in annotations_list:
            annotation["prompt_tokens"] = prompt_tokens // len(annotations_list)  # Distribute tokens evenly
            annotation["completion_tokens"] = completion_tokens // len(annotations_list)
            annotation["total_tokens"] = total_tokens // len(annotations_list)

        return annotations_list

    except Exception as e:
        print(f"Error processing batch: {e}")
        default_annotation = {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}
        return [default_annotation] * len(reviews_batch)

def validate_json(response_text: str) -> list:
    """Ensure the response is a valid JSON object with a 'reviews' array matching the expected schema."""
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
        
        # Check if response has the expected structure
        if not isinstance(response_json, dict) or 'reviews' not in response_json:
            raise ValueError("Response is missing 'reviews' key")
            
        reviews = response_json['reviews']
        if not isinstance(reviews, list):
            raise ValueError("'reviews' is not a list")

        # Validate each annotation object
        for i, annotation in enumerate(reviews):
            for emotion in expected_emotions:
                if emotion not in annotation or not isinstance(annotation[emotion], int):
                    annotation[emotion] = 0

        return reviews

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse JSON: {e}")
        print(response_text)
        return [{emotion: 0 for emotion in expected_emotions}]

def main(input_file: str, output_folder: str, batch_size: int = 5, n: int = None, model: str = "GPT-4"):
    print(f"Starting annotation process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_excel(input_file)
    if n is not None:
        df = df.head(n)

    assistant_id = get_assistant(model)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'{model}-annotations.xlsx')

    results_df = pd.DataFrame()
    total_tokens_used = 0
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0

    # Process reviews in batches
    for i in range(0, len(df), batch_size):
        batch_start_time = time.time()
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

        # Prepare batch data
        batch_data = [{"review": row['review'], "sentence": row['sentence']} for _, row in batch.iterrows()]
        
        annotations_list = get_annotation(assistant_id, batch_data)

        batch_time = time.time() - batch_start_time

        # Update results for each review in the batch
        for j, (_, row) in enumerate(batch.iterrows()):
            if j < len(annotations_list):
                annotations = annotations_list[j]
                annotations["annotation_time"] = batch_time / len(batch)
                
                total_tokens_used += annotations.get("total_tokens", 0)
                total_prompt_tokens_used += annotations.get("prompt_tokens", 0)
                total_completion_tokens_used += annotations.get("completion_tokens", 0)

                row.update(annotations)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Batch {i//batch_size + 1} processed (Time: {batch_time:.2f}s, Tokens: {total_tokens_used}, Prompt Tokens: {total_prompt_tokens_used}, Completion Tokens: {total_completion_tokens_used})")
        time.sleep(10)  # Rate limiting between batches

    print(f"Total tokens used: {total_tokens_used}")
    print(f"Total prompt tokens used: {total_prompt_tokens_used}")
    print(f"Total completion tokens used: {total_completion_tokens_used}")
    print("Annotation process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emotion annotation with Assistants API')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of reviews to process in each batch')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    parser.add_argument('--model', type=str, default='GPT-4o', help='Model name for output file')
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size, args.n, args.model)
