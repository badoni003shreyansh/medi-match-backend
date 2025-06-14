from groq import Groq
from dotenv import load_dotenv
import base64

from data_extraction import extract_json

load_dotenv()

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_ai(encoded_image):
    client = Groq()
    model = "meta-llama/llama-4-scout-17b-16e-instruct" 
    query = """Explain the disease you can see in this image. If the image is of a medical report, give a 
detailed breakdown of the report in words humans can understand. Avoid medical jargon.

CRITICAL !!!: Return the result in JSON format , in the form:

    {
        "answer": <result>
    }

If unable to understand the image, return a JSON output in the form:
{
    "answer": "Image not processed"
}

Respond ONLY with the JSON object. Do NOT add any extra text or explanation. Use double quotes for all keys and values."""

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return extract_json(chat_completion.choices[0].message.content)