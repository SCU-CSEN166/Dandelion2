# Relevance Error Analysis

Current relevance evaluation:
- TP: 20
- FP: 2
- FN: 7
- TN: 2
- Accuracy: 0.7097
- Precision: 0.9091
- Recall: 0.7407
- F1: 0.8163

## Genuine false negatives

### msg_004
Message:
Nothing major, but a few people came late and we’re still waiting on one follow-up form.

Expected: relevant
Predicted: irrelevant
Score: 0.4309

Why it may have failed:
This message refers indirectly to an already-reported activity and only makes sense with prior conversation context.

### msg_012
Message:
just recommended rn, i dont think theyve gone to the clinic yet

Expected: relevant
Predicted: irrelevant
Score: 0.1798

Why it may have failed:
The wording is informal and fragmented. The referral context is established in previous messages instead of being restated here.

### msg_027
Message:
yes [PERSON] was leading. number is [PHONE]

Expected: relevant
Predicted: irrelevant
Score: 0.4832

Why it may have failed:
The message mainly contains staff/contact information and does not explicitly restate the field activity.

## Planned-event label inconsistency

The following messages are labeled relevant in the current ground truth but have status = planned:

- msg_007: "I think 9am but I’ll double check."
- msg_014: "theres one planned for thursday in [LOCATION] but the final count isnt confirmed yet"
- msg_015: "10am i think but lemme double check"
- msg_023: "next tuesday but location not confirm yet"

The current project policy treats future/planned events as irrelevant. These labels should be reviewed before using the next relevance metrics as final results.

## False positives

### msg_029
Message:
also btw who is uploading this data to github later

Expected: irrelevant
Predicted: relevant
Score: 0.6734

Likely reason:
The classifier may be reacting to task/report-style wording even though the message is project logistics, not a field activity.

### msg_031
Message:
lowkey i was losing sleep for it

Expected: irrelevant
Predicted: relevant
Score: 0.6168

Likely reason:
Informal wording may be semantically ambiguous to the zero-shot classifier without conversation context.

## Next step

1. Correct or confirm the labels for planned/future messages.
2. Rerun the same relevance evaluation.
3. Compare the new precision, recall, and F1 with the current baseline.
4. Check whether conversation context or a revised classifier reduces the genuine false negatives without increasing false positives.
