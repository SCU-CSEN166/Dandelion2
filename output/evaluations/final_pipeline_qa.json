{
  "case_count": 6,
  "passed": 5,
  "failed": 1,
  "cases": [
    {
      "id": "qa_event_01",
      "text": "Today we held a family planning outreach in Nakuru with 31 women.",
      "anonymized_text": "Today we held a family planning outreach in Nakuru with 31 women.",
      "relevance": {
        "label": "relevant",
        "score": 0.9844125285744667,
        "uncertain": false
      },
      "extraction": {
        "status": "completed",
        "activity_type": "family planning outreach",
        "activity_description": "Today we held a family planning outreach in Nakuru with 31 women.",
        "location": "Nakuru",
        "participant_count": 31,
        "referrals": null,
        "needs_review": []
      },
      "duplicate_of": null,
      "needs_review": false,
      "review_reasons": [],
      "checks": {
        "relevance": true,
        "activity_type": true,
        "participant_count": true
      },
      "passed": true
    },
    {
      "id": "qa_social_01",
      "text": "Are we still getting tea on Saturday?",
      "anonymized_text": "Are we still getting tea on Saturday?",
      "relevance": {
        "label": "irrelevant",
        "score": 0.0,
        "reason": "question"
      },
      "extraction": null,
      "duplicate_of": null,
      "needs_review": false,
      "review_reasons": [],
      "checks": {
        "relevance": true
      },
      "passed": true
    },
    {
      "id": "qa_pii_01",
      "text": "Faith Achieng can be reached at 0712 345 678.",
      "anonymized_text": "[PERSON] can be reached at [PHONE].",
      "relevance": {
        "label": "irrelevant",
        "score": 0.4914671778678894,
        "uncertain": true
      },
      "extraction": null,
      "duplicate_of": null,
      "needs_review": true,
      "review_reasons": [
        "Verify that personal information was fully redacted.",
        "Review the relevance decision; its score is near the classification cutoff."
      ],
      "checks": {
        "contains_masks": true
      },
      "passed": true
    },
    {
      "id": "qa_screening_01",
      "text": "We held a cervical cancer screening camp with 18 women today.",
      "anonymized_text": "We held a [LOCATION] with 18 women today.",
      "relevance": {
        "label": "relevant",
        "score": 0.8979614526033401,
        "uncertain": false
      },
      "extraction": {
        "status": null,
        "activity_type": null,
        "activity_description": null,
        "location": "LOCATION",
        "participant_count": 18,
        "referrals": null,
        "needs_review": []
      },
      "duplicate_of": null,
      "needs_review": true,
      "review_reasons": [
        "Verify that personal information was fully redacted."
      ],
      "checks": {
        "relevance": true,
        "participant_count": true,
        "activity_phrase_should_remain": false
      },
      "passed": false
    },
    {
      "id": "qa_duplicate_01",
      "text": "Today we trained 24 women on kitchen gardening in Baringo.",
      "anonymized_text": "Today we trained 24 women on kitchen gardening in Baringo.",
      "relevance": {
        "label": "relevant",
        "score": 0.9882561936974525,
        "uncertain": false
      },
      "extraction": {
        "status": null,
        "activity_type": "agriculture training",
        "activity_description": "trained 24 women on kitchen gardening",
        "location": "Baringo",
        "participant_count": 24,
        "referrals": null,
        "needs_review": []
      },
      "duplicate_of": null,
      "needs_review": false,
      "review_reasons": [],
      "checks": {},
      "passed": true
    },
    {
      "id": "qa_duplicate_02",
      "text": "Twenty-four women attended today's kitchen garden training in Baringo.",
      "anonymized_text": "Twenty-four women attended today's kitchen garden training in Baringo.",
      "relevance": {
        "label": "relevant",
        "score": 0.9757402315735817,
        "uncertain": false
      },
      "extraction": {
        "status": "completed",
        "activity_type": "agriculture training",
        "activity_description": "kitchen garden training",
        "location": "Baringo",
        "participant_count": 24,
        "referrals": null,
        "needs_review": [
          "participant_count"
        ]
      },
      "duplicate_of": "qa_duplicate_01",
      "needs_review": true,
      "review_reasons": [
        "Verify the extracted participant count.",
        "Check whether this is a duplicate of qa_duplicate_01."
      ],
      "checks": {
        "duplicate_of": true
      },
      "passed": true
    }
  ]
}
