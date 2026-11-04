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
- **Problem:** What specific task are you solving?
  - The specific task we are solving with Dandelion Africa is using WhatsApp text and transcribed audio to share field updates. Our group aims to build an AI pipeline, which we want to eventually integrate into a WhatsApp bot, that converts these WhatsApp messages into organized records and removes personal information for privacy purposes.
- **Intended users:** Who would use the system, and for what decisions or activities?
  - Dandelion Africa staff would use the system to review and organize field reports more quickly and support reporting. Staff should definitely verify records before relying on them.
- **Inputs and outputs:** What will the system receive and produce? Give one concrete example.
  - The system will recieve WhatsApp text and audio, and it will transcribe the audio first. It will produce structured records with specific details such as activity type, date, location, participant count, services provided, referrals, and outcomes when stated in the text/audio.
  - Example: “Today we held a reproductive health session in Nakuru with 35 women. Three women were referred to the clinic.” → Activity: reproductive health session; Location: Nakuru; Participants: 35; Referrals: 3.
- **Success criteria:** What would make the system useful and reliable?
  - The system would be useful and reliable if if is accurate. This means it should be able to extract the stated details properly, work with text and audio which will be transcribed, and provide source evidence for the staff to review. It should flag things it is uncertain about and not assume everything is reliable, so errors can be prevented while processing the data. 
- **Constraints and risks:** [e.g., privacy, missing data, hallucinations, language coverage, human verification]
  - The biggest constraint we are facing is the fact that we are unable to use Dandelion Africa's real WhatsApp data, so in order to fix this, we are creating realistic synethic examples. Since we are creating them and they are synthetic, they may not fully reflect real messages. There are certain things we may not be able to replicate such as background noise, possible informal words being used, varied accents, etc. The system could also miss things such as names, phone numbers and personal details. Human review will definitely be necessary to ensure that everything makes sense and is accurate.

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
| Synthetic WhatsApp-style text messages | Created by the team | Testing relevance classification, anonymization, extraction, and duplicate/event detection | Start with about 100-200 English messages | Fake data only and no real private information |
| Synthetic WhatsApp audio | Recorded by team members or volunteers | Test speech-to-text and audio input | Start with about 20-40 short English clips | Using our invented content only |
| Public English accent speech data | Hugging Face/public dataset | Test transcription on different accents and speakers | English with multiple speakers/accents | Following dataset license and rules set for using it |
| AI-generated paraphrases | Created from existing synthetic examples | Add a lot more wording variation to the dataset | Depends on the starting size of the dataset | Before we use the examples, we review them | 

- **Data preparation:** We are going to create relevant and irrelevant messages. We will add fake PII like names, phone numbers, and addresses/locations. We are going to include duplicates, incomplete messages, and planned vs activities are that complete. We plan to label each example with its correct answer. We are going to keep the correct transcripts for the audio examples. Lastly, we will split the data into the development/training set and the test sets, starting with an 80/20 split. 
- **Data limitations:** The Synthetic data might not match the real Dandelion messages exactly. The starting dataset is pretty small. We might not be able to get data of every accent, writing style, or available background noises. After testing, we might have to add more examples.  

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
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
| Risha Desai | Data collection & analysis | Collect and test data from HuggingFace, Claude/AI, and team-made | Continue building out dataset with realistic data with audio. |
| Priyanka Ganguly | TTS from audio in dataset | Evaluate the speech dataset and models using precision, recall, and T1 | Experiment with other TTS models and evaluate their performance. |
| Prableen Kakar | Classification and anonymization | Evaluate dataset using anonymization and classification models | Experiment with different models on different datasets and sources. |
| Oliver Ortiz | Extraction & Integration | Primary Python implementation, input, and output, and addition work on extraction | Build out complete end to end pipeline of work. |

## Week 6 — Build: Working Baseline

Considerations
knowledgator/gliner-pii-base-v1.0

End Considerations



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


