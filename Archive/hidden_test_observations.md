Week 8 Hidden Test Notes 
These are my notes from looking through the 10-hidden. test audio messages before we were given the TA ground truth. 

msg01: detected as an event, family planning outreach, 42 participants, and the phone number was hidden.
msg03: detected as not relevant. this was mostly casual/logistics conversation and the phone number was hidden.
msg05: detected as an event, menstrual hygiene session, 60 participants, and the phone number was hidden. 
msg06: detected as an event, gender-based violence dialogue, 35 participants, and phone number hidden.
msg08: detected as not relevant. most of it was personal and social conversation but the phone number was still hidden. 
msg09: detected as an event, agriculture training, with 18 participants. 

check on these:
msg04: This message talked about a completed table banking meeting that had 25 members and 4 loans, the pipeline marked it as not relevant. Because of this, we didn't extract the activity type or participant count. This made me wonder about the relevance model and how it handles health related activties compared to activties related to finance or daily life. 
msg01 and msg02: These looked like they were describing the same family planning outreach. Both mentioned: 42 women, 6 referrals, and the same general outreach. The pipeline didn't mark either one as a duplicate though. 
msg07 and msg10: These two also looked like they could be two reports of the same event. Both of them mentioned cervical cancer screening, 28 women, and 3 referrals. The pipeline didn't end up marking these as duplicates either. 

ASR observations:
I also noticed some transcription mistakes in the audio results. Ex: "Rongai" became "wrong eye," some names were turned into English looking words or phrases. "Queue" became "key." These are important because a transcription mistake might also affect what happens later in anonymization or extraction. 

After the TA ground truth: for each message we will compare relevance, activity type, participant count, anonymization, and duplicate result. Then I can calculate the actual metrics instead of just looking at the outputs manually. 

Main things to improve after this test: 
1. Improve duplicate detection for messages that describe the same event. 
2. Test relevance on more types of Dandelion activities, not just the health activities.
3. Keep testing ASR on Kenyan/African-accented English, especially names, locations, and phone numbers.
4. Re-run the full pipeline after changes so we can see whether a fix in one part of the pipeline affects another. 