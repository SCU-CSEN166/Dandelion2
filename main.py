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


    return ConfusionMatrix(len(true_positives), len(false_positives), len(false_negatives), len(true_negatives))



class ConfusionMatrix:
    def __init__(self, TP, FP, FN, TN):
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
    
    def __add__(self, other):
        return ConfusionMatrix(self.TP+other.TP, self.FP+other.FP, self.FN+other.FN, self.TN+other.TN)
    
    def __str__(self):
        return f"TP: {self.TP}  FP: {self.FP}   FN: {self.FN}   TN: {self.TN}"
    
    def recall(self):
        return self.TP/(self.TP+self.FN)
    
    def precision(self):
        return self.TP/(self.TP+self.FP)
    
    def F1(self):
        p = self.precision()
        r = self.recall()
        
        return 2*p*r/(p+r)


def analyze_data(data: dict):
    if not data.get("records", None):
        raise RuntimeError("data not in correct format!")
    
    ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

    matrix = ConfusionMatrix(0,0,0,0)

    print(f"Loaded {len(data["records"])} records!")
    
    for record in data["records"]:        
        message = record.get("translation_en", None) or record.get("text")
        
        result = ner(message)
        
        result_matrix = score(result, record)
        
        matrix = matrix + result_matrix
        
        
        print(result_matrix)

    return matrix
    

with open("synthetic_unanonymized.json", "r") as file:
    data = json.loads(file.read())


matrix = analyze_data(data)

print(f"Total Score: {matrix}")
print(f"{matrix.recall() = }    {matrix.precision() = }     {matrix.F1() =}")