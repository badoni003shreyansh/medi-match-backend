import requests
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize model & ChromaDB client
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="chroma_db_store")
collection = client.get_or_create_collection(name="clinical_trials")


def fetch_trials_helper(query, location, max_trials=50):
    """
    Fetch trials from ClinicalTrials.gov and store in ChromaDB.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": query, "query.locn": location, "pageSize": max_trials}

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    studies = data.get("studies", [])
    trials_data = []

    for study in studies:
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        ##desc_module = protocol.get("descriptionModule", {})

        trial_id = id_module.get("nctId", "")
        title = id_module.get("briefTitle", "")
        description = ""

        if trial_id and title:
            trials_data.append({"id": trial_id, "title": title, "description": description})

    # Store trials in ChromaDB
    trial_texts = [trial["title"] + ". " + trial["description"] for trial in trials_data]
    trial_embeddings = model.encode(trial_texts)

    collection.add(
        ids=[str(trial["id"]) for trial in trials_data],
        documents=trial_texts,
        metadatas=[{"title": trial["title"], "description": trial["description"]} for trial in trials_data],
        embeddings=trial_embeddings.tolist()
    )

    print(f"Fetched & stored {len(trials_data)} trials in ChromaDB.")
    return trials_data


def find_matching_trials(age, gender, diagnosis, medications, allergies, location, k=3):
    """
    Match patient profile to top k trials using ChromaDB.
    """
    try:
        fetch_trials_helper(query=diagnosis, location=location)
        if isinstance(medications, str):
            medications = [m.strip() for m in medications.split(',') if m.strip()]
        if isinstance(allergies, str):
            allergies = [a.strip() for a in allergies.split(',') if a.strip()]
        try:
            profile_text = (
                f"{age}-year-old {gender} diagnosed with {diagnosis}. "
                f"Medications: {', '.join(medications)}. Allergies: {', '.join(allergies)}."
            )
        except:
            print("DEBUG TYPES:", type(medications), medications)
            print("DEBUG TYPES:", type(allergies), allergies)


        profile_embedding = model.encode(profile_text).tolist()
        results = collection.query(query_embeddings=[profile_embedding], n_results=k)

        matches = []
        for trial_id, metadata in zip(results["ids"][0], results["metadatas"][0]):
            matches.append({"id": trial_id, "title": metadata["title"], "description": metadata["description"]})

        return matches
    except ValueError:
        return ValueError

'''
if __name__ == "__main__":
    # Step 1: Fetch & Store Trials
    fetch_trials_helper("lung cancer", "Delhi")

    # Step 2: Find Matching Trials for a Patient Profile
    patient_profile = {
        "age": 45,
        "gender": "female",
        "diagnosis": "hypertension",
        "medications": ["lisinopril"],
        "allergies": ["penicillin"]
    }

    matches = find_matching_trials(**patient_profile)
    print("\nMatching Trials:", matches)
'''


