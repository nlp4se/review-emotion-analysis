import google.generativeai as genai
import json
import os
import argparse
from dotenv import load_dotenv
from annotate_emotions import EmotionAnnotator

# Load environment variables
load_dotenv()

class GeminiEmotionAnnotator(EmotionAnnotator):
    def _initialize_client(self) -> genai:
        """Initialize Gemini client with API key."""
        genai.configure(api_key=self.api_key)
        return genai

    def get_annotation(self, reviews_batch: list, **kwargs) -> tuple:
        """Get emotion annotations for a batch of reviews using Gemini API."""
        model = kwargs.get('model', 'gemini-pro')
        temperature = kwargs.get('temperature', 0)
        
        try:
            # Configure the model
            model = self.client.GenerativeModel(model_name=model,
                                              generation_config={
                                                  "temperature": temperature,
                                                  "top_p": 1,
                                                  "top_k": 1
                                              })
            
            # Prepare the prompt
            user_prompt = json.dumps(reviews_batch, ensure_ascii=False)
            
            # Get response from model
            response = model.generate_content(user_prompt)
            
            # Parse annotations
            annotations_list = self.validate_json(response.text)
            
            # Collect usage statistics (Note: Gemini might not provide detailed token usage)
            usage_metadata = {
                'prompt_tokens': getattr(response, 'prompt_token_count', 0),
                'completion_tokens': getattr(response, 'completion_token_count', 0),
                'total_tokens': getattr(response, 'total_token_count', 0)
            }

            return annotations_list, usage_metadata

        except Exception as e:
            print(f"Error processing batch: {e}")
            default_annotation = {emotion: 0 for emotion in self.expected_emotions}
            default_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            return [default_annotation] * len(reviews_batch), default_usage

def main():
    parser = argparse.ArgumentParser(description='Run emotion annotation with Google Gemini API')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--batch_size', type=int, default=5, 
                       help='Number of reviews to process in each batch')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    parser.add_argument('--model', type=str, default='gemini-pro', 
                       help='Model name to use')
    parser.add_argument('--temperature', type=float, default=0, 
                       help='Temperature for model responses (0-1)')
    parser.add_argument('--sleep_time', type=float, default=1, 
                       help='Sleep time between batches')
    
    args = parser.parse_args()
    
    annotator = GeminiEmotionAnnotator(api_key=os.getenv("GOOGLE_API_KEY"))
    annotator.annotate(
        input_file=args.input,
        output_folder=args.output,
        batch_size=args.batch_size,
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        sleep_time=args.sleep_time
    )

if __name__ == "__main__":
    main() 