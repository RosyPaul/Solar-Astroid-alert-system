from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_explanation(flare_data: dict, prediction: int, probability: float):
    status = "HAZARDOUS" if prediction == 1 else "SAFE"
    
    prompt = f"""
    A solar flare has been detected and analyzed:
    - Flare Class: {flare_data['flare_class']}{flare_data['flare_intensity']}
    - Source Location: {flare_data.get('sourceLocation', 'Unknown')}
    - Active Region: {flare_data.get('activeRegionNum', 'Unknown')}
    - Linked Space Events: {"Yes" if flare_data['has_linked_events'] else "No"}
    - Detected at: {flare_data.get('beginTime', 'Unknown')}
    
    Prediction: {status} ({probability * 100:.1f}% probability)
    
    In 3-4 simple sentences explain what this flare means, why it was predicted {status},
    and what impact it could have on Earth. Keep it simple for a general audience.
    """
    
    response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents=prompt
)
    return response.text

if __name__ == "__main__":
    test_flare = {
        'flare_class': 'M',
        'flare_intensity': 9.3,
        'sourceLocation': 'N13W10',
        'activeRegionNum': 14455.0,
        'has_linked_events': 1,
        'beginTime': '2026-06-03T01:22Z'
    }
    explanation = generate_explanation(test_flare, prediction=1, probability=0.826)
    print(explanation)