from transformers import pipeline
import json
from dataclasses import dataclass



def overlaps(prediction, entity):
    # True if the two spans share at least one character
    return prediction["start"] < entity["end"] and entity["start"] < prediction["end"]


def was_caught(entity, predictions):
    return any(overlaps(prediction, entity) for prediction in predictions)


def score(predictions, record):
    should_redact = [e for e in record["entities"] if e["should_redact"]]
    should_keep = [e for e in record["entities"] if not e["should_redact"]]

    # PII the model found
    true_positives = [e for e in should_redact if was_caught(e, predictions)]

    # PII the model missed (privacy leaks)
    false_negatives = [e for e in should_redact if not was_caught(e, predictions)]

    # Safe text the model correctly left alone
    true_negatives = [e for e in should_keep if not was_caught(e, predictions)]

    # Predictions that didn't hit any real PII
    false_positives = [p for p in predictions if not any(overlaps(p, e) for e in should_redact)]

    return {
        "TP": len(true_positives),
        "FP": len(false_positives),
        "FN": len(false_negatives),
        "TN": len(true_negatives),
    }




def analyze_data(data: dict):
    if not data.get("records", None):
        raise RuntimeError("data not in correct format!")
    
    ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

    total_score = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    print(f"Loaded {len(data["records"])} records!")
    
    for record in data["records"]:        
        message = record.get("translation_en", None) or record.get("text")
        
        result = ner(message)
        
        score_result = score(result, record)
        print(score_result)
        
        for entry in total_score:
            total_score[entry] += score_result[entry]
        
        
        # print(result)

    return total_score
    

with open("synthetic_unanonymized.json", "r") as file:
    data = json.loads(file.read())


total_score = analyze_data(data)

print(f"Total Score: {total_score}")