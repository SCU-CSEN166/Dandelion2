# Week 8 Hidden-Test TA Comparison and Extraction Analysis

## Purpose

This file compares the team's latest hidden-test pipeline output (`output/HIDDEN_TEST_OUTPUT.csv`) against the TA-provided ground truth (`hidden_test_outputs.csv`). It focuses on relevance classification, extraction of activity type and participant count, and duplicate detection.

## Summary metrics

### Relevance classification

- True positives: 8
- False positives: 2
- False negatives: 0
- True negatives: 0
- Accuracy: 0.800 (80.0%)
- Precision: 0.800 (80.0%)
- Recall: 1.000 (100.0%)
- F1: 0.889

The updated relevance setup found all 8 true Events, so recall reached 1.00. However, it also classified both TA Chit-chat messages (`msg03` and `msg08`) as Events. This created two false positives and lowered overall relevance accuracy to 80%.

### Participant-count extraction

- Correct participant counts on TA Event messages: 8/8
- Participant-count accuracy on events: 1.000 (100.0%)

All 8 TA Event messages had the correct participant count in the latest pipeline output. This includes `msg04`, which now correctly reached extraction and produced 25 participants after the relevance change.

### Activity-type extraction

- Exact string matches to the TA activity label: 2/8 (25.0%)
- Exact matches occurred for `msg01` and `msg02` ("Family planning outreach").
- The remaining Event messages generally captured the correct activity topic but used broader or less-specific wording.

Because the extraction model generates free-text activity labels, exact-string matching is conservative. Examples:
- `msg04`: TA = "Women's group table banking (savings) meeting"; pipeline = "Savings group meeting"
- `msg05`: TA = "Menstrual hygiene session and sanitary pad distribution"; pipeline = "Menstrual hygiene session"
- `msg06`: TA = "Gender-based violence awareness dialogue"; pipeline = "Gender-based violence dialogue"
- `msg07`/`msg10`: TA = "Cervical cancer screening camp"; pipeline = "Cancer screening"
- `msg09`: TA = "Kitchen garden training"; pipeline = "Agriculture training"

The pipeline therefore captured the general activity for each Event, but several outputs were not as specific as the TA label.

### Duplicate detection

- Correct duplicate status overall: 8/10 (80.0%)
- True duplicate reports in TA ground truth: 2
- True duplicates correctly detected by pipeline: 0/2

The two missed duplicate reports were:
- `msg02` should be `duplicate_of = 1`
- `msg10` should be `duplicate_of = 7`

The latest pipeline output marked both as `none`, so duplicate-event detection remains an important gap.

## Message-by-message comparison

| Message | TA relevance | Pipeline relevance | TA activity | Pipeline activity | TA participants | Pipeline participants | TA duplicate | Pipeline duplicate |
|---|---|---|---|---|---:|---:|---|---|
| msg01 | Event | Event | Family planning outreach | Family planning outreach | 42 | 42 | none | none |
| msg02 | Event | Event | Family planning outreach | Family planning outreach | 42 | 42 | 1 | none |
| msg03 | Chit-chat | Event | — | Other | — | — | none | none |
| msg04 | Event | Event | Women's group table banking (savings) meeting | Savings group meeting | 25 | 25 | none | none |
| msg05 | Event | Event | Menstrual hygiene session and sanitary pad distribution | Menstrual hygiene session | 60 | 60 | none | none |
| msg06 | Event | Event | Gender-based violence awareness dialogue | Gender-based violence dialogue | 35 | 35 | none | none |
| msg07 | Event | Event | Cervical cancer screening camp | Cancer screening | 28 | 28 | none | none |
| msg08 | Chit-chat | Event | — | Community health visit | — | — | none | none |
| msg09 | Event | Event | Kitchen garden training | Agriculture training | 18 | 18 | none | none |
| msg10 | Event | Event | Cervical cancer screening camp | Cancer screening | 28 | 28 | 7 | none |

## Important findings

### 1. Relevance tradeoff after the Week 8 change

The updated relevance configuration fixed the previous `msg04` miss: the table-banking report is now classified as an Event and its participant count is extracted correctly.

However, the broader treatment of plans also caused `msg03` and `msg08` to become false positives. Both contain social plans but are Chit-chat according to the TA ground truth. This shows that improving recall by broadly counting plans as relevant can reduce precision.

### 2. Extraction is strong for numeric participant counts

Once a true Event reaches the extraction stage, participant-count extraction was correct on all 8 TA Event messages in this hidden set.

This also demonstrates why relevance errors matter: in the earlier run, `msg04` was filtered out before extraction. Once relevance allowed it through, the extractor correctly returned 25 participants.

### 3. Activity labels need more specificity

The extractor usually identified the correct general topic, but several activity labels were broader than the TA ground truth. A future improvement could use a controlled activity taxonomy or normalization step so semantically equivalent outputs map to one standard label.

### 4. Duplicate detection is still unresolved

Both true duplicate reports were missed. This cannot be solved reliably by exact text matching because the duplicate reports use different wording. A stronger approach should compare event-level features such as activity type, participant count, referrals, location, and surrounding context.

## Priority improvements

1. Refine relevance so planned field activities can count as relevant without treating ordinary social plans as Events.
2. Add an activity-type normalization/taxonomy step so extraction produces consistent labels.
3. Implement event-level duplicate detection using extracted fields and semantic/context similarity.
4. Continue end-to-end evaluation because an upstream relevance error determines whether extraction runs at all.

## Evidence files

- TA ground truth: `hidden_test_outputs.csv`
- Team prediction: `output/HIDDEN_TEST_OUTPUT.csv`
- Row-by-row comparison: `output/hidden_test_ta_comparison.csv`
