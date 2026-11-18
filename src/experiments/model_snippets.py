"""Small trials of the candidate models we tried, kept for reference. Not part of the pipeline.

    uv run -m src.experiments.model_snippets               # every trial (downloads every model)
    uv run -m src.experiments.model_snippets flan_t5 gliner
"""

import argparse
import json
from datetime import datetime

SAMPLE_AUDIO = "Archive/sample.opus"
SAMPLE_REPORT = "Done! 28 women served at Mogotio, 3 referred"
MAX_NEW_TOKENS = 128


def distil_whisper():
    from transformers import pipeline
    speech_to_text = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
    print(speech_to_text(SAMPLE_AUDIO)["text"])


def whisper_multilingual():
    from transformers import pipeline
    speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    print(speech_to_text(SAMPLE_AUDIO, generate_kwargs={"language": "english"})["text"])


def bart_zero_shot():
    """Also tried: MoritzLaurer/deberta-v3-large-zeroshot-v2.0."""
    from transformers import pipeline
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print(zero_shot("Done! 28 women served, 3 referred",
                    candidate_labels=["completed activity report", "planned future activity", "unrelated chat"]))


def bert_ner():
    from transformers import pipeline
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    print(ner("Grace Wanjiku held a session in Nakuru with 35 women"))


def gliner():
    from gliner import GLiNER
    model = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
    print(model.predict_entities("Wanjiru, 34, call me on 0711 000 123", ["name", "age", "phone number"],
                                 threshold=0.3))


def flan_t5():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    question = tokenizer("Where were the women served? Text: Done! 28 women served at Mogotio", return_tensors="pt")
    answer = model.generate(**question, max_new_tokens=20)
    print(tokenizer.decode(answer[0], skip_special_tokens=True))


def nuextract():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_name = "numind/NuExtract-2.0-2B"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True)
    template = json.dumps({"activity": ["family planning", "youth session", "savings group", "training"],
                           "location": "verbatim-string", "participants": "integer", "referrals": "integer"})
    prompt = processor.tokenizer.apply_chat_template([{"role": "user", "content": SAMPLE_REPORT}],
                                                     template=template, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    print(processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])


def minilm_similarity():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(["Mogotio session today 28 mothers", "Done! 28 women served at Mogotio"])
    print(model.similarity(embeddings[0], embeddings[1]))


def dates_and_number_words():
    import dateparser
    from word2number import w2n
    print(dateparser.parse("yesterday", settings={"RELATIVE_BASE": datetime(2026, 11, 2, 13, 47)}))  # 2026-11-01
    print(w2n.word_to_num("thirty five"))  # 35


TRIALS = {trial.__name__: trial for trial in [
    distil_whisper, whisper_multilingual, bart_zero_shot, bert_ner, gliner, flan_t5, nuextract,
    minilm_similarity, dates_and_number_words,
]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="*", choices=list(TRIALS), help="default: all of them")
    for name in parser.parse_args().trials or TRIALS:
        print(f"== {name}")
        TRIALS[name]()


if __name__ == "__main__":
    main()
