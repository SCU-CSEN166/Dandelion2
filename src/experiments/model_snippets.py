"""Scratch trials of candidate models (ASR, zero-shot, NER, Flan-T5, MiniLM,
Whisper, GLiNER, NuExtract), kept for reference and the "models tried" write-up.
Not part of the pipeline: running it downloads every model.

    uv run -m src.experiments.model_snippets
"""

import json
from datetime import datetime

import dateparser
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoProcessor,
                          AutoTokenizer, pipeline)
from word2number import w2n

SAMPLE_AUDIO = "Archive/sample.opus"
SAMPLE_REPORT = "Done! 28 women served at Mogotio, 3 referred"
MAX_NEW_TOKENS = 128

# SPEECH TO TEXT
speech_to_text = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
print(speech_to_text(SAMPLE_AUDIO)["text"])

# LABELING (also tried: MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zero_shot("Done! 28 women served, 3 referred",
                candidate_labels=["completed activity report", "planned future activity", "unrelated chat"]))

# ANONYMIZER
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Grace Wanjiku held a session in Nakuru with 35 women"))

# EXTRACTION
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
question = flan_tokenizer("Where were the women served? Text: Done! 28 women served at Mogotio", return_tensors="pt")
answer = flan_model.generate(**question, max_new_tokens=20)
print(flan_tokenizer.decode(answer[0], skip_special_tokens=True))

# SIMILARITY
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = sentence_model.encode(["Mogotio session today 28 mothers", "Done! 28 women served at Mogotio"])
print(sentence_model.similarity(embeddings[0], embeddings[1]))

# LANGUAGE ALT
multilingual_speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small")
print(multilingual_speech_to_text(SAMPLE_AUDIO, generate_kwargs={"language": "english"})["text"])

# PII
gliner_pii = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
print(gliner_pii.predict_entities("Wanjiru, 34, call me on 0711 000 123", ["name", "age", "phone number"], threshold=0.3))

# DATES AND NUMBER WORDS
print(dateparser.parse("yesterday", settings={"RELATIVE_BASE": datetime(2026, 11, 2, 13, 47)}))  # 2026-11-01 13:47
print(w2n.word_to_num("thirty five"))  # 35

# BETTER EXTRACTION
nuextract_name = "numind/NuExtract-2.0-2B"
processor = AutoProcessor.from_pretrained(nuextract_name, trust_remote_code=True, padding_side="left")
nuextract = AutoModelForImageTextToText.from_pretrained(nuextract_name, trust_remote_code=True)
template = json.dumps({"activity": ["family planning", "youth session", "savings group", "training"],
                       "location": "verbatim-string", "participants": "integer", "referrals": "integer"})
prompt = processor.tokenizer.apply_chat_template([{"role": "user", "content": SAMPLE_REPORT}],
                                                 template=template, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], return_tensors="pt")
output = nuextract.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
prompt_length = inputs["input_ids"].shape[1]
print(processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0])
