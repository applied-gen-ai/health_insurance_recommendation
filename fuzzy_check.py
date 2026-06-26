from policy import insurance_data
from rapidfuzz import process, fuzz

def fuzzy_match(d):
    # Step 1: Match provider (keys)
    provider_result = process.extractOne(d["provider"], insurance_data.keys(), scorer=fuzz.ratio)

    if provider_result:  # Ensure a match was found
        provider_match = provider_result[0]  # Best match
        provider_score = provider_result[1]  # Match score
    else:
        return {"error": "No provider match found"}

    # Step 2: Match insurance name (values under the matched provider)
    policy_list = insurance_data.get(provider_match, [])
    policy_result = process.extractOne(d["insurance_name"], policy_list, scorer=fuzz.ratio)

    if policy_result:
        policy_match = policy_result[0]  # Best match
        policy_score = policy_result[1]  # Match score
    else:
        return {"error": "No insurance match found"}

    return {
        "matched_provider": provider_match,
        "provider_match_score": provider_score,
        "matched_insurance_name": policy_match,
        "insurance_match_score": policy_score
    }

