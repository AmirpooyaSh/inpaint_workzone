import os
import sys
import json
import base64
from openai import OpenAI
from typing import List, Dict, Any

# --- Tool Definition ---
# Define the function and its parameters that the LLM will be forced to call.
# This structure ensures the output is always in the desired format.
FUNC_SAFETY_REPORT = {
    "type": "function",
    "function": {
        "name": "report_safety_observation",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "",
                    "items": {
                        "type": "object",
                        "properties": {
                            "worker_name": {
                                "type": "string",
                                "description": "",
                            },
                            "safety_issue": {
                                "type": "string",
                                "description": "",
                            },
                        },
                        "required": ["worker_name", "safety_issue"],
                    },
                }
            },
            "required": ["observations"],
        },
    },
}


def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file to a base64 string for API transmission.

    Args:
        image_path: The file path to the image.

    Returns:
        A base64 encoded data URL string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Basic check for image type based on extension
        mime_type = f"image/{image_path.split('.')[-1]}"
        return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

class AI_Agent:
    """
    An agent to send multimodal prompts to an LLM and force a structured response.
    NOTE: The 'o3' model must support vision capabilities.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "o3", # Assumes 'o3' is a multimodal model endpoint
        base_url: str = "https://api.openai.com/v1",
    ):
        """Initializes the OpenAI client."""
        # Use provided API key or get it from environment variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("API key not found. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
            sys.exit(1)

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model

    def send_multimodal_prompt(self, text_prompt: str, image_path: str) -> List[str] | None:

        base64_image = encode_image_to_base64(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        }
                    ],
                }],
                tools=[FUNC_SAFETY_REPORT],
                tool_choice={"type": "function", "function": {"name": "report_safety_observation"}}
            )

            message = response.choices[0].message
            tool_call = message.tool_calls[0] if message.tool_calls else None

            if tool_call:
                # The arguments are returned by the API as a JSON string
                arguments = json.loads(tool_call.function.arguments)
                print("✅ Model returned the following data:")
                print(json.dumps(arguments, indent=2))
                
                # Format the response as a list per the requirements
                return [
                    arguments.get("worker_name", "N/A"),
                    arguments.get("identical_information", "N/A"),
                    arguments.get("safety_issue", "N/A"),
                ]
            else:
                print("⚠️  The model did not return the expected tool call.")
                return None

        except Exception as e:
            print(f"An error occurred while communicating with the API: {e}")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create an instance of the agent
    #    Make sure your API key is set as an environment variable `UFL_API_KEY`
    #    or pass it directly: AI_Agent(api_key="your_key_here")
    agent = AI_Agent()

    # 2. Define the prompt and the path to your image
    prompt_text = "What are the safety issue(s) associated with the non-machinery operator workers in the image?"
    
    # IMPORTANT: Replace this with the actual path to your image file
    image_file_path = "partial_replacement_3.jpg"

    # 3. Send the prompt and get the structured response
    if not os.path.exists(image_file_path):
        print("="*50)
        print(f"🚨 ERROR: The image file was not found at the specified path.")
        print(f"Please update the 'image_file_path' variable in the script to a valid path.")
        print(f"Current path: '{image_file_path}'")
        print("="*50)
    else:
        observation_data = agent.send_multimodal_prompt(
            text_prompt=prompt_text,
            image_path=image_file_path
        )

        # 4. Print the final, formatted list
        if observation_data:
            print("\n--- Final Extracted Information (as a list) ---")
            print(observation_data)