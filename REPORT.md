# Assignment 2 Report

- **Name**: Yasmin Tandon
- **Student ID**:
- **Email**: ytandon.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**The decision.** On a commercial poultry farm, staff walk each house daily and
look at fresh droppings, because a dropping's colour and consistency are among the
earliest visible signs of the three diseases that matter most in a flock:
**coccidiosis**, **salmonellosis** and **Newcastle disease (NCD)**. Today that judgement
is made by eye by farm workers with uneven training. The model takes a phone photo
of one dropping sample and returns one of three outcomes:

- **Pass:** looks healthy, no action.
- **Fail:** suspected disease. A confirmatory PCR test is sent and the vet is called.
- **Human review:** uncertain. A trained vet technician looks at the photo and decides pass or fail.

**Who acts on it.** Farm staff take the photos on the daily walk-through. A vet
technician works the review queue. The farm vet acts on every fail.

**Why the two errors cost different amounts.**
- **A false negative** (a diseased sample passed) lets an infectious disease keep spreading
  through a house of thousands of birds until someone notices it later. It costs mortality,
  lost weight gain, a larger treatment bill, and for NCD a notifiable-disease response.
- **A false positive** (a healthy sample failed) costs one PCR test and a vet visit that finds nothing.

The miss is worse by more than an order of magnitude. That asymmetry drives everything below:
- The model is tuned for **disease recall**.
- Thresholds sit **low**.
- The review band exists to catch uncertain cases cheaply, before they become either error.

**Cost per sample, as assumed in `train.py`:**

| Outcome | Cost (EUR) | Reasoning |
|---|---|---|
| False negative: diseased sample passed | **2,000** | One extra day of undetected spread in one house before the next sampling round catches it: extra mortality and lost growth across the house, plus a larger treatment course. |
| False positive: healthy sample failed | **80** | A confirmatory PCR panel plus the farm vet's time on a visit that finds nothing. |
| True positive: diseased sample failed | **80** | The same PCR and visit. This is the action we want, but it is not free. |
| True negative: healthy sample passed | 0 | No action. |
| Human review of one sample | **2** | 3 minutes of a vet technician at EUR 40/hour. The reviewer is assumed to decide correctly. |

A miss therefore costs **25 times** a false alarm. These are assumptions, not
measured figures. The sensitivity analysis below shows that the chosen
thresholds hold for a miss cost anywhere between EUR 500 and EUR 2,000. They only
move once a miss is assumed to cost about EUR 5,000 or more.

## Dataset

- **Roboflow Universe:** workspace `thesis-pr4oh`, project `chicken-droppings-analysis`, **version 1**
- **Link:** https://universe.roboflow.com/thesis-pr4oh/chicken-droppings-analysis/dataset/1
- **License:** CC BY 4.0. Used as published, not forked, with the native train/valid/test split.
- **Task:** object detection with 4 classes: `Coccidiosis-like`, `Healthy`, `NCD-like`, `Salmonella-like`.
- **What one image is:** a phone photo of one dropping on a farm floor (concrete, litter, soil),
  mostly 288×640 or 640×288 pixels. That is the same kind of photo the deployed
  system would receive.
- **Export settings:** auto-orient, then resize to fit within 640×640. Roboflow augmented
  the training split only, with 3 outputs per source image: horizontal and vertical flips,
  90° rotations, shear of ±10° and exposure of ±10%.

**Why it serves the decision rather than just resembling it:**
- **Same unit:** one image is one dropping sample, exactly what farm staff would photograph.
- **Same classes:** they are the three diseases the farm needs to act on, plus healthy.
- **Same capture conditions:** real floors with real clutter, not studio shots.

**One caveat:** the `-like` suffix suggests the labels are **visual diagnoses, not
lab-confirmed ones**. The project page does not document how the labels were made.
The model can therefore at best reproduce an expert's visual call, which is exactly
why every fail is confirmed by PCR.

## What the dataset's composition implies

| Split | Images | Source images | Coccidiosis | Healthy | NCD | Salmonella |
|---|---|---|---|---|---|---|
| train | 13,644 | 4,548 (each ×3 augmented) | 4,308 | 4,915 | **891** | 4,875 |
| valid | 1,298 | 1,298 | 412 | 473 | **95** | 457 |
| test | 648 | 648 | 212 (197 img) | 232 (205 img) | **51 (38 img)** | 223 (208 img) |

Counts are box instances; test image counts are in brackets. The split is about
70/20/10 by source image. Grouping files on the name stem before `.rf.` shows that
no source image appears in more than one split. Valid and test are not augmented.

**What this implies:**

1. **NCD results cannot be trusted to a precise number.**
   - NCD has 5–6× fewer instances than the other classes, and only **38 test images (51 boxes)**.
   - One image more or less moves NCD image recall by 2.6 points.
   - The 95% Wilson interval on NCD box recall (35/51 = 0.69) is **0.55–0.80**.
   - NCD is also the most dangerous disease to miss. So the class we care most about is the one we can measure least precisely.
2. **It is effectively image classification.**
   - 13,640 of 13,644 training images contain exactly one class, with 1.1 boxes per image on average.
   - The decision unit is therefore one image, and the cost analysis scores each image by its highest disease-class confidence.
3. **There are no background-only images.**
   - Every image contains a labelled dropping, so the test set says nothing about false alarms on empty floor, feathers or feed.
   - In production, a photo with no dropping in it is possible and untested.
4. **Box size goes with class.**
   - Median box area as a share of the image: **Healthy ~6%**, NCD ~9–19%, Salmonella ~20%, **Coccidiosis ~42%**.
   - The model can partly use size as a shortcut. If farm staff photograph droppings from a different distance than the dataset's photographers did, accuracy may drop.
5. **The class mix is nothing like a farm.**
   - 68% of test images (443/648) are diseased, against an assumed farm base rate of 5%.
   - Raw test accuracy or precision would therefore overstate how often a fail is real. The cost analysis reweights to the base rate instead of using the test mix.
6. **Photos were collected class by class** (filenames start with the class name).
   - Within a class, photos may share a farm, a floor or lighting, which the test split would not separate.
   - Test performance may therefore be optimistic for a new farm.

## Finetuning and baselines

**Model and task.**
- **Model:** `yolo11s`, finetuned from the COCO-pretrained checkpoint, as object detection.
- **Why not `yolo11n`:** Healthy droppings are small (~6% of the image), and the larger model
  handles small objects better at modest extra cost.
- **Why detection rather than classification:** detection is how the dataset is labelled, and
  boxes show the reviewer which object drove the call.

**Training setup** (all values in `train.py`):
- **Hardware:** Kaggle notebook, **Tesla T4 GPU**. 30 epochs took **1.36 hours**.
- **Image size:** 640, which matches the export, so no further downscaling.
- **Epochs:** at most 30, with early stopping patience 5. Early stopping never triggered.
- **Optimiser:** AdamW, lr0 = 0.001, batch 16, seed 0, deterministic.
- **Augmentation:** Ultralytics defaults on top of Roboflow's. Mosaic is switched off for the last 10 epochs.
- **Loss:**
  - box (CIoU) gain 7.5 and DFL gain 1.5, unchanged
  - the classification term replaced by **0.5 × (0.7 × BCE + 0.3 × Focal)**, with focal γ = 1.5 and α = 0.25
  - The focal component down-weights easy examples (mostly background locations and the
    well-separated large classes), shifting gradient towards hard and rare cases such as NCD.
  - BCE stays dominant so that confidence scores remain usable for thresholding.
- **Committed weights:** the best checkpoint by validation fitness, as `weights/best.pt`.

**Results on the same test split (648 images):**

| Model | mAP50 | mAP50-95 | Disease recall | Expected cost / 1,000 samples |
|---|---|---|---|---|
| **Finetuned yolo11s** (pass < 0.04, fail ≥ 0.71) | **0.896** | **0.523** | **99.5%** | **EUR 4,535** |
| COCO yolo11s, zero-shot | 0.000 | 0.000 | 0% | EUR 100,000 |
| Trivial: predict nothing (pass everything) | 0.000 | 0.000 | 0% | EUR 100,000 |
| Trivial: majority class (Salmonella-like; 32.1% accuracy) | n/a | n/a | 100% | EUR 80,000 |

**What the gaps show:**
- **COCO zero-shot** has no dropping or disease class, so it cannot match a single label.
  It still fires on 227 of 648 test images at confidence 0.25, with irrelevant COCO
  classes, which shows that an object detector seeing *something* is not the same as answering the question.
  For the farm it is equivalent to passing everything.
- **Majority class** fails every sample, which catches every disease but pays for a PCR and a vet visit
  on every healthy sample.
- **The finetuned model beats all three by a wide margin** on both accuracy and cost. The gap
  is larger than the gap between the two trivial baselines themselves.
- **Validation mAP** rose from 0.71 after epoch 1 to about 0.91 by epoch 22, then flattened (0.905–0.911).
  It was still edging up at epoch 30, so a longer run may add a little.

## Per-class evaluation and failure modes

**Positive class for costing:** *diseased*, meaning any of Coccidiosis, NCD or Salmonella.
Each image's score is its highest disease-class confidence.

**Detection metrics on test** (Ultralytics; precision and recall at its max-F1 confidence):

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| Coccidiosis-like | 0.918 | 0.929 | 0.946 | 0.607 |
| Healthy | 0.882 | 0.906 | 0.920 | 0.501 |
| **NCD-like** | 0.946 | **0.682** | **0.773** | 0.433 |
| Salmonella-like | 0.904 | 0.919 | 0.943 | 0.550 |
| **All** | | | **0.896** | **0.523** |

**Image-level confusion matrix** at the single-threshold operating point (0.13). Each image takes its
top detection; below the threshold it counts as "none".

| true \ predicted | Coccidiosis | Healthy | NCD | Salmonella | none |
|---|---|---|---|---|---|
| Coccidiosis-like (197) | **197** | 0 | 0 | 0 | 0 |
| Healthy (205) | 0 | **202** | 0 | 2 | 1 |
| NCD-like (38) | 0 | 2 | **34** | 2 | 0 |
| Salmonella-like (208) | 1 | 3 | 0 | **203** | 1 |

**Where the headline comes from, and where the model breaks:**
- **Coccidiosis and Salmonella carry the headline number.** They are the two largest classes,
  with AP50 of about 0.94–0.95 each.
- **NCD drags it down.**
  - Box recall is 0.68 and AP50 is 0.77, the lowest on every metric.
  - At image level 34 of 38 NCD images are right. Two are called Healthy and two Salmonella.
  - Two NCD images have disease scores of **0.002 and 0.007**: the model sees no disease
    at all. No sensible threshold catches them, and they are the only misses at the chosen review band.
- **NCD is also the least confident class.**
  - Median disease score: NCD 0.64, against 0.74 for Coccidiosis and 0.76 for Salmonella.
  - Only 32% of NCD images score above the fail threshold of 0.71, against 64–69% for the other diseases.
  - So NCD leans on human review far more than the other diseases do.
- **At box level, the largest confusion is true Coccidiosis predicted as Salmonella (40 of 212 boxes).**
  - At image level this mostly disappears, because the top detection in those images is still Coccidiosis.
  - For the farm the confusion matters less: both are diseases and both lead to a fail and a PCR test, which names the pathogen.
- **Healthy has the lowest precision (0.88)**, partly because its boxes are smallest and the hardest to localise.

Training curves and Ultralytics' PR, F1, precision and recall curves are in `outputs/training/` and `outputs/test_eval/`.

## Cost matrix and operating point

**Method.**
- Each test image gets a disease score.
- For each threshold, pass/review/fail rates are measured **separately on diseased (443) and healthy (205) test images**,
  then combined at the assumed **5% base rate**.
- Cost per sample = 0.05 × (cost on diseased) + 0.95 × (cost on healthy), using the cost table above.
- Thresholds are swept from 0.00 to 1.00 in steps of 0.01 (`outputs/cost_sweep.json`, plotted in the dashboard).

**Why 5%:** droppings are sampled across healthy houses every day. On most days most
houses are clean, and a sample is only diseased while an outbreak is under way.
5% is an assumption, not a measured prevalence.

| Operating point | Thresholds | Cost / 1,000 | Disease recall | Healthy failed | Sent to review |
|---|---|---|---|---|---|
| Model alone (minimum of single-threshold sweep) | 0.13 | EUR 6,196 | 98.9% | 1.5% | n/a |
| **Model + review band (minimum over all pairs)** | **pass < 0.04, fail ≥ 0.71** | **EUR 4,535** | **99.5%** | **0.0%** | **5.1%** |

**Why the single-threshold optimum sits so low (0.13):** a miss costs 25× a false alarm,
so the sweep accepts 3 healthy samples failed out of 205 to miss only 5 diseased out of 443.
The five misses are the 2 near-zero NCD images and three Salmonella images scoring 0.05–0.12.

**Sensitivity: what moves the answer.**

| Assumption varied | Range tested | Chosen band | Cost / 1,000 |
|---|---|---|---|
| Base rate | 0.5% → 10% | stays **0.04 / 0.71** | EUR 515 → 9,001 |
| Base rate | **20%** | **0.00 / 0.71**: review nearly everything below 0.71 (87% of samples) | EUR 17,747 |
| Cost of a miss | EUR 500 → 2,000 | stays 0.04 / 0.71 | EUR 4,196 → 4,535 |
| Cost of a miss | EUR 200 | 0.09 / 0.71 | EUR 4,123 |
| Cost of a miss | **EUR 10,000** | **0.00 / 0.71**: review 97% | EUR 5,937 |
| Cost of a false alarm | EUR 20 → 500 | stays 0.04 / 0.71 | EUR 1,548 → 25,440 |
| Review time | 1 → 30 min | stays 0.04 / 0.71 | EUR 4,467 → 5,446 |

**The assumption that would most change the answer is the base rate, together with the cost of a miss.**
- The band is stable from 0.5% to 10% prevalence and from EUR 500 to EUR 2,000 per miss.
- If prevalence rises towards **20%** (an outbreak season), or a miss costs **EUR 10,000 or more**
  (for example an NCD case triggering movement restrictions), the cheapest design is to send almost
  every sample below 0.71 to a person. At that point the model is only used to fast-track
  confident fails.
- The false-alarm cost and the review time change the total but not the thresholds.

## What a person still has to check

- **Review band:** 0.04 ≤ disease score < 0.71 goes to a person.
- **Share sent to review:** **5.1%** of samples at a 5% base rate.
  - On the test set the band contains 161 of 443 diseased images and 7 of 205 healthy ones.
  - The band holds a large share of diseased samples (36%), because NCD and some Salmonella
    score in the middle range. At the farm's 5% base rate that is still only about 1 sample in 20.
- **Load at an assumed 500 samples a day** (for example 10 houses × 50 samples):
  - about **25 reviews a day**, at **3 minutes** each, is about **76 minutes a day**
  - that is **0.17 of one person's working day** (7.5 hours)
- **This is realistic.** One vet technician can clear the queue in a single sitting alongside
  other duties, and the load stays under half a person even at 10% prevalence (6.7% of samples sent to review).

**The design against the three alternatives** (5% base rate):

| Design | Cost / 1,000 samples | People needed | Comment |
|---|---|---|---|
| Do nothing (pass everything) | EUR 100,000 | 0 | every diseased sample is missed |
| Person checks every sample | EUR 6,000 | 3.3 (1,500 min/day) | perfect accuracy is assumed, so this is a lower bound on its real cost |
| Model decides alone (threshold 0.13) | EUR 6,196 | 0 | 5 misses and 3 false alarms per 648 test images |
| **Model + human review band** | **EUR 4,535** | **0.17** | 2 misses, 0 false alarms on test |

**Reading the table:**
- **The model alone is slightly *more* expensive than a person checking everything.** Its
  few misses are costly, and review is cheap at EUR 2.
- **Its value comes from deciding where people look.** The review band saves 24% against
  checking everything, and cuts staffing from 3.3 people to a fraction of one.
- **Against doing nothing**, it saves about EUR 95 per sample.
- **The comparison with "person checks everything" flatters the human**, because it assumes a perfect reviewer.
  A real reviewer misses some cases, which would make the model + review design look better still.

## Limitations & next steps

1. **NCD is thin and weakest.** There are 38 test images, and box recall is 0.69 (95% CI 0.55–0.80).
   - **Next:** add NCD images from other sources or farms, and evaluate NCD recall on a larger held-out set.
   - Also compare against a class-weighted classification loss on NCD.
2. **Two NCD samples score close to zero.** No threshold catches them.
   - **Next:** inspect them to see whether the label or the image is at fault.
   - If the image is at fault, this is a real blind spot to report to the vet.
3. **No background or non-dropping images.** False alarms on empty floor, feathers or feed are untested.
   - **Next:** collect or add background images, and include them in the test set.
4. **Box size correlates with class**, so the model may partly learn a shortcut.
   - **Next:** test on photos taken from a different distance, and use scale augmentation if performance drops.
5. **Labels are visual (`-like`), not lab-confirmed.**
   - **Next:** validate against PCR-confirmed samples before using the model's score as a disease probability.
6. **Possible farm or session effects.** Photos were collected class by class, and the test split is not separated by farm.
   - **Next:** evaluate on photos from a farm that is not in the dataset.
7. **The costs, base rate and review accuracy are assumptions.** The reviewer is assumed to be perfect.
   - **Next:** replace them with the farm's real PCR price, the vet's rate and historical outbreak losses.
   - The dashboard already accepts new cost and base-rate figures live.
8. **Training was capped at 30 epochs.** Validation mAP was still rising slightly, and early stopping never fired.
   - **Next:** a longer run (for example 60 epochs, patience 10) with the same settings, compared on the same test split.

## Generative AI use disclosure

Claude (Anthropic, via Claude Code) was used to:
- search Roboflow Universe for candidate datasets
- download the dataset and compute its composition statistics
- write `train.py`: the download, the training call, the custom BCE + focal classification loss, evaluation, baselines and the cost sweep
- write `app.py`, the Gradio dashboard
- run training on a Kaggle GPU
- draft this report, including the per-sample cost assumptions and the sensitivity analysis

I chose the dataset, the model (`yolo11s`), the image size, the training length,
the optimiser and learning rate, the batch size and seed, and the loss design and its parameters.
