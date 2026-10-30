# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D2
**Challenge:** Dandelion
**Team members:** Prableen, Priyanka, Risha, Oliver
**Repository:** https://github.com/SCU-CSEN166/Dandelion2
**Last updated:** 10-29-26

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations
[Keep a current list of what the system cannot reliably do.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact
- **Problem:** What specific task are you solving? [Write a precise problem statement.]
- **Intended users:** Who would use the system, and for what decisions or activities?
- **Inputs and outputs:** What will the system receive and produce? Give one concrete example.
- **Success criteria:** What would make the system useful and reliable?
- **Constraints and risks:** [e.g., privacy, missing data, hallucinations, language coverage, human verification]

### 2. Proposed technical approach
- **Pipeline:**
1) Start with WhatsApp text or audio messages.

2) Convert audio messages into text.

3) Look for personal information like names, phone numbers, locations, etc, and anonymize it.

4) Check whether each message has useful info about field activities or is just regular conversation messages.

5) For the relevant messages, pull out important details like what happened, where it happened, when it happened, and how many people were involved.

6) Compare messages to check if any are talking about the same activity.

7) Put everything into a structured record/database that someone can review and fix if needed.

- **Baseline(s):**

  We're planning to use pretrained models and see how well they work on the messages we create. Since we do not have access to the real dataset and are planning to create our own synthetic data, we are starting with pretrained models instead of training our own from scratch. We're going to start by converting audio messages to text and checking whether each message is relevant. We're also going to try to remove any kind of personal info before extracting the details that we need. For now, we're mainly trying to get these steps working together. Once we have something running, we can look at where it's making mistakes and improve those parts.

- **Proposed models/tools:** 

distil-whisper/distil-small.en (https://huggingface.co/distil-whisper/distil-small.en) - Speech-to-text model that translates spoken English into written words. We want to use this to turn audio messages into text. We chose a smaller Whisper model because it should be easier to run with the resources we have.

facebook/bart-large-mnli (https://huggingface.co/facebook/bart-large-mnli) - Can sort text into categories without needing to be trained on our own examples first. We're going to use this for figuring out whether a message is relevant or irrelevant.

dslim/bert-base-NER (https://huggingface.co/dslim/bert-base-NER) - Finds things like people's names, organizations, locations and miscellaneous entities in text. We're going to use this to look for names and locations that need to be anonymized. We'll also look for phone numbers and email addresses using patterns (phone numbers usually have that hyphen pattern and emails have patterns like something@xyz.com) in case the AI model misses them.

google/flan-t5-base (https://huggingface.co/google/flan-t5-base) - Text-to-text model that can follow instructions and give answers based on the text that is inputted. We're going to use this for pulling out information like activity type, date, location, and participant count from the relevant messages.

sentence-transformers/all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Compares how similar two messages are. We want to see if this can help us find messages that are talking about the same event even if they're worded differently.

- **Alternative approach:**

  If the classification model isn't doing well, we can try labeling some of our synthetic messages ourselves and using them to fine-tune a smaller model to help it recognize the kinds of messages we're working with.
   For anonymization, we're concerned that the NER model might miss some names or personal details, especially if the messages have spelling mistakes or are written informally, but we can try adding more rules for those cases and compare a few different models if needed. We also have to make sure we're not taking out any important information.
   If the extraction model doesn't work with longer or unclear messages, we can lessen the number of fields and use easier rules where possible. But it will be easier to figure out what needs to change once we've tested the whole pipeline on some examples so the plan we have right now is a good starting point that can be edited later.

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| Team-created synthetic Dandelion WhatsApp messages | Created by the team | Main development and evaluation data for relevance classification, extraction, event grouping, and deduplication | Initial target: ~100-200 messages; English | Synthetic or anonymized, no real private information |
| Team records synthetic WhatsApp audio | Created by team members | Speech-to-text testing and realistic audio inputs | Initial target: ~20-40 clips in English | Recordings will use invented field activity content, no private conversations included |
| Public English accent dataset | Hugging Face or public dataset | Test speech-to-text performance across different English speaking accents | multiple English speakers/accents | Has to follow dataset license and the rules |
| AI-generated paraphrasing or synthetic messages | Generated from existing synthetic examples | Data augmentation and testing variation in the wording | Dependent on the starting dataset size | Generated examples will be reviewed by team before using | 

- **Data preparation:** We will create a set of realistic Dandelion kind of messages based on the types of field activities the organization has described on their website, like health outreach, youth programs, livelihood programs for women, trainings, and other community activities. We will include relevant and irrelevant messages so that the classifier has realistic examples. We also will intentionally have repeated reports, incomplete messages, messages about planning, and those about the same activity but written in a different format. Each message will have ground-truth labels from the team. The labels could include: relevant or irrelevant, activity type, date, location, participant count, service or outcome, if personal information is present, whether the message is a duplicate or related to a different message. For audio examples, we will keep the correctly written transcript so we can compare the model transcription against the ground truth we chose. We will divide the data into separate development and test sets so we don't evaluate only on examples we used while designing the system. A possible starting split would be 80% development/training data and 20% held-out testing data. 
- **Data limitations:** Our largest limitation is that our synthetic data might not be a perfect representation of real Dandelion Africa WhatsApp conversations. Real messages could have other vocabulary, accents, background noise, or communication patterns that we aren't able to reproduce. Our starting dataset could be too small too. Therefore, we need to treat the first dataset as a starting point and then expand while we learn which types of examples are too difficult for the system. 

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| [Metric - e.g Accuracy, F1, AoC] | [Quality dimension] | [Method/ Formula] | [Target] |
|Recall| Can be used for anonymization as we will want to know how good we are at completely anonomyzing a source | TP/(TP+FN) where TP is the number of successful anonymizations, and FN is data that should be anonymized but wasn't. | Ideally we want as close to 100% accuracy as possible because we do not want to let personal information into our data.
| Precision | Of the data we extracted, how much was correct and not extracted accidentally | TP / (TP+FP) where TP is data successfully extracted and FP is data extracted that should not have been extracted | We want this to perform better than trying to trivially match keywords in the text
| F1 | Can use to determine how well we classify activities in messages | 2pr/(p+r) where p = precision and r = recall | We will want this to perform significantly better than checking for just checking for certain keywords in the messages
| Confusion matrix | Shows the True Positive, False Positive, True Negative, and False Negative results from our classifer, such as anonymization and extraction | Create chart with predicted class and actual class | Ideally, we want TP and TN to be much higher than FP and FN.

- **Evaluation examples:** We will use our synthetic dataset which we will know how many anonymizations are needed and what data needs to be extracted. We can run our model on that data and compare the results against the known answer.
- **Human verification:** Either our Data Collection team member or classisification and anomyization team member will oversee our test results against the expected answers from our data to determine if any of our classifiers are performing badly.
- **Failure cases to watch:** 
- One problem might be with our speech to text transcriber turning names, addresses, or phone numbers into data that will be much harder to be caught by the classifier. For example, a phone number might be transcribed as just a 9 digit number and might be missed by our anonymizer.
- Another problem might be overfitting of our models. This is especially important given that our synthetic data may not actually align perfectly with the real-world data. It will be important our models are able to adapt to new and different forms of information.

### 5. Team responsibilities
| Member | Primary responsibility | Specific Weekly contribution | Planned next task |
|---|---|---|---|
| Risha Desai | Data Collection/synthetic data | Research and create initial synthetic WhatsApp text examples and define ground-truth labels | Build on dataset and create the training and testing split |
| Priyanka Ganguly | Speech-to-text and audio | Research speech datasets and proposed Whisper baseline, while also collecting or creating sample audio clips | Implement Distil-Whisper baseline and continue evaluating sample transcriptions |
| Prableen Kakar | Classification and anonymization | Research relevance classification and NER scores | Implement initial relevant & irrelevant classifier and anonymization baseline |
| Oliver Ortiz | Extraction + Evaluation + Integration | Define structured output and evaluation metrics and our research extraction approach | Implement starting structured extraction and connect the pipeline components |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status
- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| [Name] | [Specific contribution] | [Status] | [Link] | [Next step] |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| [Case 1] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 2] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 3] | [Observation] | [Hypothesis] | [Impact] | [Fix] |

### Changes attempted after stress-testing
| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| [Change] | [Status] | [Result] | [Decision] |

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment
- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| [Issue] | [Change] | [Status] | [Link or metric] | [Limitation] |

### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Reflection] |



### Final submission checklist
- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements
[Python version, libraries, compute requirements, model access]

### Installation
```bash
# Add installation commands here
```

### Run the system
```bash
# Add an example command here
```

### Run evaluation
```bash
# Add an example evaluation command here
```

### Repository structure
```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```


