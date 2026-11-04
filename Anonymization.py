from transformers import pipeline

# TTS
asr = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
print(asr("sample.opus")["text"])

# LABELING
# clf2 = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(clf("Done! 28 women served, 3 referred", candidate_labels=["completed activity report", "planned future activity", "unrelated chat"]))

# ANONYMIZER
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Grace Wanjiku held a session in Nakuru with 35 women"))

# EXTRACTION
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("google/flan-t5-base"); t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
ids = tok("Where were the women served? Text: Done! 28 women served at Mogotio", return_tensors="pt")
print(tok.decode(t5.generate(**ids, max_new_tokens=20)[0], skip_special_tokens=True))

# SIMILARITY
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
e = st.encode(["Mogotio session today 28 mothers", "Done! 28 women served at Mogotio"]); print(st.similarity(e[0], e[1]))

# LANGUAGE ALT
asr2 = pipeline("automatic-speech-recognition", model="openai/whisper-small")
print(asr2("sample.opus", generate_kwargs={"language": "english"})["text"])

# PII
from gliner import GLiNER
pii = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
print(pii.predict_entities("Wanjiru, 34, call me on 0711 000 123", ["name", "age", "phone number"], threshold=0.3))

# DATES
import dateparser; from datetime import datetime
print(dateparser.parse("yesterday", settings={"RELATIVE_BASE": datetime(2026, 11, 2, 13, 47)}))  # 2026-11-01 13:47
from word2number import w2n; print(w2n.word_to_num("thirty five"))  # 35

# BETTER EXTRACTION
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
name = "numind/NuExtract-2.0-2B"
proc = AutoProcessor.from_pretrained(name, trust_remote_code=True, padding_side="left")
nx = AutoModelForImageTextToText.from_pretrained(name, trust_remote_code=True)
tmpl = json.dumps({"activity": ["family planning", "youth session", "savings group", "training"], "location": "verbatim-string", "participants": "integer", "referrals": "integer"})
text = proc.tokenizer.apply_chat_template([{"role": "user", "content": "Done! 28 women served at Mogotio, 3 referred"}], template=tmpl, tokenize=False, add_generation_prompt=True)
inp = proc(text=[text], return_tensors="pt"); out = nx.generate(**inp, max_new_tokens=128, do_sample=False)
print(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])




"""
TODO: Presenting Week 6
1) Parse WhatsApp data
2) Transcribe when necessary
3) Anonymize Data
4) Determine Relevance
5) Extract to JSON


# AND SETUP GUIDE




# Test data (synthetic, human, WhatsApp, AfriSpeech data from Hugginface)
# Input that (capable of accepting WhatsApp primarily)
# GO THROUGH PIPELINE:
    1) Parse WhatsApp data
    2) Transcribe when necessary
    3) Anonymize Data
    4) Determine Relevance
    5) Extract to JSON
# Show processed and filtered output (in like a .json)
# Show results from confusion matrix where applicable
# Install guide
# Models used, models tried
# Attempts
# Contributions




Running Baseline &
Results: metrics: accuracy, recall, 

"""
