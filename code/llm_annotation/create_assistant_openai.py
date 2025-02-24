import os
import argparse
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv
from create_assistant import AssistantCreator

class OpenAIAssistantCreator(AssistantCreator):
    def _initialize_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)
    
    def get_json_schema(self) -> Dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "classify_review_emotions",
                "description": "Classifies a list of app review sentences into emotion categories, returning a structured JSON array.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reviews": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Joy": { "type": "integer", "enum": [0, 1] },
                                    "Trust": { "type": "integer", "enum": [0, 1] },
                                    "Fear": { "type": "integer", "enum": [0, 1] },
                                    "Surprise": { "type": "integer", "enum": [0, 1] },
                                    "Sadness": { "type": "integer", "enum": [0, 1] },
                                    "Disgust": { "type": "integer", "enum": [0, 1] },
                                    "Anger": { "type": "integer", "enum": [0, 1] },
                                    "Anticipation": { "type": "integer", "enum": [0, 1] },
                                    "Neutral": { "type": "integer", "enum": [0, 1] },
                                    "Reject": { "type": "integer", "enum": [0, 1] }
                                },
                                "required": ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral", "Reject"]
                            }
                        }
                    },
                    "required": ["reviews"]
                }
            }
        }
    
    def create_assistant(self, guidelines_file: str, model: str = "gpt-4") -> Any:
        guidelines = self.load_guidelines(guidelines_file)
        instructions = self.get_base_instructions(guidelines)
        
        print("Creating a new OpenAI assistant...")
        
        assistant = self.client.beta.assistants.create(
            name="Emotion Annotation Assistant",
            instructions=instructions,
            model=model,
            temperature=0.0,
            response_format=self.get_json_schema()
        )
        
        self.save_assistant_id(assistant.id, model)
        print(f"Assistant created successfully. Assistant ID: {assistant.id}")
        return assistant

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Create an OpenAI assistant for emotion annotation.')
    parser.add_argument('--guidelines', required=True, help='Path to the guidelines file')
    parser.add_argument('--model', default="gpt-4o", help='OpenAI model to use')
    
    args = parser.parse_args()
    
    creator = OpenAIAssistantCreator(api_key=os.getenv("OPENAI_API_KEY"))
    creator.create_assistant(args.guidelines, args.model)

if __name__ == "__main__":
    main() 