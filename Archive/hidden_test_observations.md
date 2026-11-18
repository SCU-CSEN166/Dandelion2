# Week 8 Hidden-Test Observations

This file records preliminary observations from the 10-message hidden stress test before TA ground-truth metrics are available.

## Current hidden-test run

The system produced outputs for all 10 hidden voice messages.

Current pipeline output file:
- `output/reports.csv`

## Preliminary successes

### msg01
- Classified as Event
- Activity type: Family planning outreach
- Participants: 42
- Phone number was masked

### msg03
- Classified as Not relevant
- Casual/logistics conversation was filtered out
- Phone number was masked

### msg05
- Classified as Event
- Activity type: Menstrual hygiene session
- Participants: 60
- Phone number was masked

### msg06
- Classified as Event
- Activity type: Gender-based violence dialogue
- Participants: 35
- Phone number was masked

### msg08
- Classified as Not relevant
- Social/personal conversation was filtered out
- Phone number was masked

### msg09
- Classified as Event
- Activity type: Agriculture training
- Participants: 18
- Phone number was masked

## Preliminary failure cases

### msg04 — likely relevance failure
The message reports a completed table-banking meeting with 25 members and four loans.

Current output:
- Relevance: Not relevant
- Activity type: none
- Participants: none

Likely issue:
The relevance classifier may generalize better to health/outreach examples than livelihood or finance activities such as table banking.

Potential impact:
A real field activity could be dropped before extraction.

### msg01 and msg02 — likely duplicate event pair
Both messages describe the same family-planning outreach:
- 42 women
- 6 referrals
- same outreach context

Current output:
- `duplicate_of = none` for both

Likely issue:
The current pipeline does not yet perform event-level duplicate detection.

### msg07 and msg10 — likely duplicate event pair
Both messages describe the same cervical-cancer screening:
- 28 women screened
- 3 referrals
- same screening context

Current output:
- `duplicate_of = none` for both

Likely issue:
The current pipeline does not yet compare extracted event details across messages.

## Cross-pipeline observations

Some transcription errors changed names, locations, or ordinary words. Examples include:
- "Rongai" becoming "wrong eye" in one transcript
- some names becoming English-looking phrases
- "queue" becoming "key"

These errors can propagate into later stages such as anonymization and extraction.

## What to check once TA ground truth is available

For every message, compare:
- relevance label
- activity type
- participant count
- anonymization
- duplicate status

Then calculate:
- relevance accuracy, precision, recall, and F1
- activity-type accuracy on events
- participant-count accuracy on events
- anonymization success / unnecessary masking
- duplicate-status accuracy

## Prioritized next improvements

1. Add event-level duplicate detection using extracted fields and message context.
2. Expand relevance examples to cover more Dandelion activity types, including livelihood and financial activities.
3. Continue improving ASR on Kenyan/African-accented English so upstream transcription errors do not propagate.
4. Re-test the full pipeline after each component change instead of evaluating only individual models.
