import requests
from groq import Groq
from data_extraction import extract_json
from dotenv import load_dotenv

load_dotenv()

def extract_summary(data):
        protocol = data.get("protocolSection", {})
        id_mod = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        design_mod = protocol.get("designModule", {})
        arms_mod = protocol.get("armsInterventionsModule", {})
        outcome_mod = protocol.get("outcomesModule", {})
        desc_mod = protocol.get("descriptionModule", {})
        cond_mod = protocol.get("conditionsModule", {})

        title = id_mod.get("officialTitle", id_mod.get("briefTitle", "N/A"))
        status = status_mod.get("overallStatus", "N/A")
        phase = ", ".join(design_mod.get("phases", []))
        conditions = ", ".join(cond_mod.get("conditions", []))
        purpose = design_mod.get("designInfo", {}).get("primaryPurpose", "N/A")

        arms = arms_mod.get("armGroups", [])
        interventions = []
        for arm in arms:
            label = arm.get("label", "N/A")
            desc = arm.get("description", "N/A")
            interventions.append(f"{label}: {desc}")
        intervention_text = "; ".join(interventions)

        outcomes = outcome_mod.get("primaryOutcomes", [])
        primary_outcomes = "; ".join([o.get("measure", "") for o in outcomes])

        summary = desc_mod.get("briefSummary", "N/A")

        return f"""
Title: {title}
Status: {status}
Phase: {phase}
Conditions: {conditions}
Primary Purpose: {purpose}
Interventions: {intervention_text}
Primary Outcomes: {primary_outcomes}
Summary: {summary}
        """.strip()

def fetch_clinical_trial_info(nct_ids):

    all_contexts = []
    for nct_id in nct_ids:
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data for NCT ID {nct_id}. Status code: {response.status_code}")

        data = response.json()
        trial_summary = extract_summary(data)
        all_contexts.append(trial_summary)

    return "\n\n---\n\n".join(all_contexts)

import json
import re
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def clean_llm_json(response_content):
    """
    Safely cleans and parses LLM output which may contain extra backticks, 'json' tags, or unwrapped multiple objects.
    """
    # Remove backticks and 'json' tag
    cleaned = response_content.strip().strip('`').replace('json', '', 1).strip()

    # If already starts with [ we assume valid JSON array
    if cleaned.startswith('['):
        try:
            return json.loads(cleaned)
        except Exception as e:
            print("JSON parsing failed even after cleaning:", e)
            raise e

    # If not wrapped in array, attempt to extract objects and wrap them
    try:
        objects = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
        array_text = "[" + ",".join(objects) + "]"
        return json.loads(array_text)
    except Exception as e:
        print("JSON recovery failed:", e)
        raise e


def ask_ai_about_trial(context_text, patient_age, patient_gender, patient_diagnosis, patient_medication, patient_allergies):
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=[
            "context_text", "patient_age", "patient_gender",
            "patient_diagnosis", "patient_medication", "patient_allergies"
        ],
        template="""
You are a medical expert AI assistant. Given a list of clinical trials and a patient profile, return a JSON array.

For each clinical trial in the context, return:
- trial_summary: What is the trial about?
- suitability: Is the patient likely eligible based on age, gender, diagnosis, etc.?
- potential_benefits: How might the patient benefit?

Inputs:

Clinical Trial Information:
{context_text}

Patient Information:
Age: {patient_age}
Gender: {patient_gender}
Diagnosis: {patient_diagnosis}
Current Medications: {patient_medication}
Allergies: {patient_allergies}

Output JSON format:
[
  {{
    "trial_id":"..."
    "trial_summary": "...",
    "suitability": "...",
    "potential_benefits": "..."
  }},
  ...
]
Respond ONLY with the JSON.
"""
    )

    # Prepare the filled prompt
    final_prompt = prompt_template.format(
        context_text=context_text,
        patient_age=patient_age,
        patient_gender=patient_gender,
        patient_diagnosis=patient_diagnosis,
        patient_medication=patient_medication,
        patient_allergies=patient_allergies
    )

    # Load Gemini model via LangChain

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
    )

    # Call Gemini with the prepared prompt
    response = llm.invoke(final_prompt)

    # Extract text content from response
    response_content = response.content
    # Clean and parse JSON
    data = clean_llm_json(response_content)

    return data
