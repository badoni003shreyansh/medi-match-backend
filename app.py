import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from clinical_trials_ai import ask_ai_about_trial, fetch_clinical_trial_info
from image_ai_helper import analyze_image_with_ai, encode_image
from trial_matcher_model import find_matching_trials

app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files['image']
    encoded_image = encode_image(image)
    answer = analyze_image_with_ai(encoded_image)
    return jsonify({"answer": answer})
    
@app.route("/fetch_clinical_trials",methods=["POST"])
def fetch_clinical_trials():
    data = request.get_json()

    age = data.get("age")
    gender = data.get("gender")
    diagnosis = data.get("diagnosis")
    medications = data.get("medications")
    allergies = data.get("allergies")
    location = data.get("location")

    # Find matching trials
    matching_trials = find_matching_trials(age, gender, diagnosis, medications, allergies, location, k=3)
    
    if not matching_trials:
        return "No matching trials found."

    # Use first trial
    nctID1 = matching_trials[0]['id']
    nctID2= matching_trials[1]['id']
    nctID3=matching_trials[2]['id']

    nctID = [nctID1, nctID2, nctID3]

    # Fetch full clinical trial data (ensure it's in string format)
    clinical_trials_data = fetch_clinical_trial_info(nctID)

    # If clinical_trials_data is dict, convert to text for LLM
    if isinstance(clinical_trials_data, dict):
        clinical_trials_text = json.dumps(clinical_trials_data, indent=2)
    else:
        clinical_trials_text = clinical_trials_data

    # Call Gemini model
    answer = ask_ai_about_trial(
        clinical_trials_text,
        age, gender, diagnosis, medications, allergies
    )

    return jsonify({"answer": answer})


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"SERVER LISTENING AT PORT: {port}")
    app.run(port=port)

    