import pandas as pd
import openai
import json
import os
from typing import Dict
import time
import argparse

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_guidelines(guidelines_path: str) -> str:
    """Load annotation guidelines from text file."""
    with open(guidelines_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_prompt(review: str, sentence: str, guidelines: str) -> str:
    """Create the system and user prompts."""
    system_prompt = """You are an expert emotion annotator. Analyze the review and classify emotions 
    according to the provided guidelines. Return your annotations in JSON format with exactly these 
    emotion categories: joy, trust, fear, surprise, sadness, disgust, anger, anticipation, neutral, mixed. 
    Use 1 for presence and 0 for absence of each emotion.
    
    Guidelines:
    {guidelines}
    """
    
    user_prompt = json.dumps({
        "review": review,
        "sentence": sentence
    }, ensure_ascii=False)
    
    return system_prompt.format(guidelines=guidelines), user_prompt

def get_gpt4_annotation(review: str, sentence: str, guidelines: str) -> Dict[str, int]:
    """Get emotion annotations from GPT-4 for a single review."""
    system_prompt, user_prompt = create_prompt(review, sentence, guidelines)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using GPT-4 instead of GPT-4V
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0,  # Setting temperature to 0 for maximum consistency
            max_tokens=300
        )
        
        # Parse the response to ensure it's in the correct format
        result = json.loads(response.choices[0].message.content)
        expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
        
        # Validate and ensure all emotions are present
        annotations = {emotion: result.get(emotion, 0) for emotion in expected_emotions}
        return annotations
    
    except Exception as e:
        print(f"Error processing review: {e}")
        return {emotion: 0 for emotion in expected_emotions}

def main(input_file: str, guidelines_file: str, output_folder: str):
    """
    Process annotations with parameters from calling script.
    
    Args:
        input_file (str): Path to the input Excel file
        guidelines_file (str): Path to the guidelines text file
        output_folder (str): Path to the output directory
    """
    # Read the Excel file and guidelines
    df = pd.read_excel(input_file)
    guidelines = load_guidelines(guidelines_file)
    
    # Emotion columns
    emotion_columns = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                      'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Mixed']
    
    # Process each review
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")
        
        annotations = get_gpt4_annotation(
            review=row['review'],
            sentence=row['sentence'],
            guidelines=guidelines
        )
        
        # Update the DataFrame with annotations
        for emotion in emotion_columns:
            df.at[idx, emotion] = annotations[emotion]
        
        # Add delay to respect rate limits
        time.sleep(3)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the results
    output_file = os.path.join(output_folder, 'GPT-4o-annotations.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Annotation completed! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-4 emotion annotation')
    parser.add_argument('--input', required=True, help='Path to input Excel file')
    parser.add_argument('--guidelines', required=True, help='Path to guidelines text file')
    parser.add_argument('--output', required=True, help='Path to output folder')
    
    args = parser.parse_args()
    main(args.input, args.guidelines, args.output)
