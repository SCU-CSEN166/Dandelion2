Week 6 Test Results

Test 1: PII anonymization
Input: Grace Wanjiru is handling them. Her number is 0712 345 678.
Expected: Grace Wanjiru anonymized & 0712 345 678 anonymized
Actual: Grace Wanjiru was replaced with [PERSON] & 0712 345 678 was replaced with [PHONE]
Result: Pass
Overall anonymization evaluation
- TP: 11
- FP: 0
- FN: 1
- Precision: 1.00
- Recall: 0.9167
- F1: 0.9565
- Fully redacted PII: 11 of 12 annotated entities

Test 2: Irrelevant message
Input: yall think we're gonna get the midterm back soon🙁
Expected: Irrelevant
Actual: Exact output for this specific message still needs to be recorded.
Result: Pending
Overall relevance-classification evaluation
- Labeled records: 31
- TP: 20
- FP: 2
- FN: 7
- TN: 2
- Accuracy: 0.7097
- Precision: 0.9091
- Recall: 0.7407
- F1: 0.8163

Test 3: Duplicate event
Input 1: they visited 14 homes today and around 21 women received health information
Input 2: household visits done, 20+ women reached and 3 need follow up in Kapkures
Expected: Same or possible duplicate event
Actual: Duplicate-detection output has not been recorded yet.
Result: Pending

Test 4: WhatsApp audio transcription
File: 00003085-AUDIO-2026-11-04-16-05-26.opus
Ground truth: wait yeah i was like losing so much sleep like oh my god guys before tuesdays class and then we had the quiz i was tweaking out
Distil-Whisper output:
But yeah, I was like losing so much steep, like, oh my god, guys, before Tuesday's class, and then we had the quiz, I was tweaking out.
Result: Partly

The transcription almost captured everything, but it changed the opening phrase to "but yeah" instead of "wait year" and transcribed “sleep” as “steep.”

More WhatsApp audio outputs
File: 00003084-AUDIO-2026-11-04-16-05-11.opus
Ground truth:lowkey me too im really stressed about what my grade is gonna be and i mean she said she's not gonna curve so im really nervous
Distil-Whisper output: Loki meets you, I'm really stressed about what my grade is going to be and I mean she said she's not going to curb so I'm really nervous.
Result: Partly

The main meaning was preserved, but informal speech wasn't transcribed correctly, including “lowkey me too,” and “curve” was outputted as “curb.”

File: 00003086-AUDIO-2026-11-04-16-05-50.opus
Ground truth: guys lets all go to office hours to learn more about about her cats and her childhood in romania
Distil-Whisper output: Guys, let's all go to office hours to learn more about her cat and her childhood in Romania.
Result: Partly
The transcription was very close, but it changed “cats” to just “cat” and removed the second “about.”

Even more speech-to-text evaluation
Distil-Whisper was also tested on 10 clips from a Nigerian accented person speaking English.
Word Error Rate (WER): 0.42296
The model performed better on the shorter and more clear clips, while other clips showed bigger errors that involved issues with names, wording, and more informal, less structured speech.

Current Week 6 takeaway
The text-processing baseline is running on exported WhatsApp messages. PII anonymization did decent but sometimes misses important, relevant messages. Relevance classification works but also misses some relevant messages. Distil-Whisper can transcribe the synthetic WhatsApp voice notes, but the informal words and a few regular individual words seem to trip up the model and get transcribed incorrectly. 
The text-processing baseline runs on exported WhatsApp messages. We still need to do duplicate-event detection and document those results. 
