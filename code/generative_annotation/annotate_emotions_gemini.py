import google.generativeai as genai
import pandas as pd
import json
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def get_chat_model(guidelines_path: str):
    """Initialize and return the Gemini model with proper configuration."""
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json",
    }

    # Load guidelines
    with open(guidelines_path, 'r', encoding='utf-8') as file:
        guidelines = file.read()

    # Format instructions similar to Mistral
    system_instruction = f"""
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

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

    return model.start_chat(history=[])

def get_annotation(chat_session, reviews_batch: list) -> tuple:
    """Classify emotions for a batch of reviews using the chat session."""
    try:
        # Convert batch to JSON string
        user_prompt = json.dumps(reviews_batch, ensure_ascii=False)
        
        # Get response from chat session
        response = chat_session.send_message(user_prompt)
        
        # Extract the content from the nested structure
        content = response.candidates[0].content.parts[0].text.strip()
        
        # Parse and validate the JSON content
        annotations_list = validate_json(content)
        
        # Extract usage metadata from the correct location
        usage_metadata = {
            'prompt_tokens': response.usage_metadata.prompt_token_count,
            'completion_tokens': response.usage_metadata.candidates_token_count,
            'total_tokens': response.usage_metadata.total_token_count
        }
        
        return annotations_list, usage_metadata

    except Exception as e:
        print(f"Error processing batch: {e}")
        default_annotation = {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}
        default_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        return [default_annotation] * len(reviews_batch), default_usage

def validate_json(response_text: str) -> list:
    """Ensure the response is a valid JSON object with a 'reviews' array matching the expected schema."""
    expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                         'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    try:
        response_json = json.loads(response_text.strip())
        
        if not isinstance(response_json, list):
            raise ValueError("Response is not a list")
        
        for annotation in response_json:
            for emotion in expected_emotions:
                if emotion not in annotation or not isinstance(annotation[emotion], int):
                    annotation[emotion] = 0

        return response_json

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return [{emotion: 0 for emotion in expected_emotions}]

def main(input_file: str, output_folder: str, batch_size: int = 5, n: int = None, model: str = "gemini-2.0-flash", guidelines_path: str = 'guidelines.txt'):
    start_time = datetime.now()
    print(f"Starting annotation process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_excel(input_file)
    if n is not None:
        df = df.head(n)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'{model}-annotations.xlsx')
    metrics_file = os.path.join(output_folder, f'{model}-metrics.json')

    results_df = pd.DataFrame()
    metrics = {
        'start_time': start_time.isoformat(),
        'batches': [],
        'total_tokens': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0
    }

    # Initialize Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    for i in range(0, len(df), batch_size):
        # Create a new chat session for each batch
        chat_session = get_chat_model(guidelines_path)
        
        batch_start_time = time.time()
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

        batch_data = [{"review": row['review'], "sentence": row['sentence']} for _, row in batch.iterrows()]
        
        annotations_list, usage_metadata = get_annotation(chat_session, batch_data)

        # Update metrics
        batch_metrics = {
            'batch_number': i//batch_size + 1,
            'batch_size': len(batch),
            'processing_time': time.time() - batch_start_time,
            'tokens': usage_metadata
        }
        metrics['batches'].append(batch_metrics)
        metrics['total_tokens'] += usage_metadata['total_tokens']
        metrics['total_prompt_tokens'] += usage_metadata['prompt_tokens']
        metrics['total_completion_tokens'] += usage_metadata['completion_tokens']

        for j, (_, row) in enumerate(batch.iterrows()):
            if j < len(annotations_list):
                row_data = row.to_dict()
                row_data.update(annotations_list[j])
                results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                
        # Save progress
        results_df.to_excel(output_file, index=False, engine='openpyxl')
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {i//batch_size + 1} processed in {batch_time:.2f}s (Tokens: {usage_metadata['total_tokens']}, Prompt Tokens: {usage_metadata['prompt_tokens']}, Completion Tokens: {usage_metadata['completion_tokens']})")
        time.sleep(1)

    # Record end time and final metrics
    end_time = datetime.now()
    metrics['end_time'] = end_time.isoformat()
    metrics['total_duration'] = (end_time - start_time).total_seconds()
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Annotation process completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {metrics['total_duration']:.2f}s)")
    print(f"Total tokens used: {metrics['total_tokens']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emotion annotation with Google Gemini 2.0 Flash')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of reviews to process in each batch')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='Model name for output file')
    parser.add_argument('--guidelines', type=str, default='guidelines.txt', help='Path to guidelines file')
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size, args.n, args.model, args.guidelines) 