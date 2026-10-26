<div align="center">

![What a Fracture Detector Really Learns: A Leak-Controlled Audit](FractureDetection/fig/banner.png)

</div>

# What a Fracture Detector Really Learns: A Leak-Controlled Audit

## 1. Introduction

This research project audits what a standard fracture classifier learns from radiographs. A ConvNeXt-Tiny model was trained on FracAtlas under a split that keeps near-duplicate images together. Its decision threshold was fixed in advance on validation data. The model was then tested once on 20,327 paediatric wrist images from GRAZPEDWRI-DX, with no retraining and no threshold change. The classifier reaches an AUC of 0.923 on FracAtlas and 0.825 on GRAZPEDWRI-DX. On first-visit exams the external AUC falls to 0.697, and sensitivity falls to 0.540. The central finding is a treatment shortcut. Within the same patient, a follow-up image with a cast scores 4.2 logits higher than one without (95% CI 4.0 to 4.3). A fracture at first presentation raises the score by only about 2.1 logits. Follow-up exams score high even without a cast. Small fractures and patients aged 16 and over are further weak spots. A random internal split reveals none of these failures.

---

## 2. Background

Many deep-learning fracture models report internal AUCs above 0.9 on plain radiographs. Detection models have been trained on paediatric wrist radiographs [7, 8], and two-stage designs target subtle scaphoid fractures [9]. Three terms frame this study.

**Leakage** occurs when near-identical images of the same limb fall on both sides of a train and test split. The test then measures recall of a seen image, not detection of an unseen fracture. FracAtlas has no patient identifiers, so leakage must be controlled by image similarity.

A **shortcut** is a feature that correlates with the label in the training data but is not the property of interest [5]. Casts, implants and image crops are common candidates in fracture imaging.

**External validation** tests a fixed model on data from another institution, population and scanner [6]. A model that relies on shortcuts often loses accuracy there. This study treats the classifier as deliberately standard. Its contribution is the audit around the model.

---

## 3. Datasets and Preprocessing

The development set is FracAtlas [1]. It contains 4,083 radiographs, and 719 of them show a fracture (17.6%). The images cover the leg (2,124), hand (1,281), a mixed group with several labelled body parts (398), hip (179) and shoulder (101). Every fractured image carries a bounding box around the fracture. FracAtlas has no patient identifier, so the pipeline groups images instead. It links any two images whose 256-bit perceptual hashes differ by at most 30 bits. It also links consecutive images that are both flagged as multi-scan and share a body part. This produces 3,828 groups, of which 178 contain more than one image. Whole groups go to a single fold, with five folds stratified by label and body part.

The external set is GRAZPEDWRI-DX [2]. It holds 20,327 paediatric wrist images from 6,091 patients. Of these, 20,215 images come from Siemens equipment and 112 from Agfa equipment. The fracture label is the dataset's `fracture_visible` field, and a blank field counts as no fracture. This gives 13,550 fractured images (66.7%). The dataset also supplies `cast`, `metal`, `initial_exam`, age, sex and AO fracture class. An **initial exam** is an image whose `initial_exam` field equals 1; all other images count as **follow-ups**. Every external interval resamples whole patients.

Prevalence differs strongly between the two sets (17.6% against 66.7%). The study therefore compares only AUC, sensitivity and specificity across them. Table 1 summarises the two sets.

<div align="center">

| | FracAtlas (development) | GRAZPEDWRI-DX (external) |
|---|---|---|
| Images | 4,083 | 20,327 |
| Fractured | 719 (17.6%) | 13,550 (66.7%) |
| Anatomy | leg, hand, hip, shoulder, mixed | paediatric wrist only |
| Equipment | mixed | Siemens (20,215), Agfa (112) |
| Grouping | perceptual-hash near-duplicates and multi-scan runs (3,828 groups) | patient (6,091 patients) |
| Role | five-fold training and audits | one-shot external test |

<b>Table 1.</b> The two datasets. Groups are kept within one fold and resampled in every bootstrap interval

</div>

Images are resized so that the longer side is 512 pixels and padded to 512 x 512. Training uses light augmentation: rotation up to 10 degrees, scaling up to 10%, small translations, and brightness and contrast changes. It uses no flips, because laterality carries information in some views.

---

## 4. Evaluation Design

The classifier is a ConvNeXt-Tiny [3] pretrained on ImageNet, with one input channel and one output logit. It trains for up to 12 epochs with AdamW (learning rate 1e-4, weight decay 0.05), a one-cycle schedule and batch size 16. Training stops early when validation AUC has not improved for 4 epochs.

Training uses five folds. For fold k, the model trains on three folds and uses fold (k+1) mod 5 for early stopping and threshold setting. It is then tested on fold k. Each FracAtlas image is therefore scored once, by the one fold model that neither trained on it nor validated on it. These out-of-fold scores support every internal result.

The **operating point** is the decision threshold, fixed before any test data is seen. For each fold model it is the 90th percentile of the scores of non-fractured validation images. This gives validation specificity of 0.899 to 0.900, approximately 90%. Each model's logits are shifted so that its threshold sits at zero. An image is **flagged** when its shifted score is zero or above. A flagged fractured image is **detected**, and an unflagged fractured image is **missed**.

The **five-fold ensemble** score is the mean of the five fold models' shifted logits. The pipeline uses it for every external image, with the cut-off of zero and no refitting. Internally the five-fold ensemble is never used, because the other four models trained or validated on each image. The internal headline AUC pools the shifted out-of-fold scores of all 4,083 images.

Every interval is a 95% percentile bootstrap with 1,000 draws that resamples whole groups. Groups are near-duplicate groups in FracAtlas and patients in GRAZPEDWRI-DX. Two exceptions use fewer draws: the resolution test (300) and the feature-probe intervals (200). Paired comparisons of two training runs reuse the same resampled groups for both. Figure 1 shows the full pipeline.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/pipeline.png" alt="Pipeline">
<br>
<b>Figure 1.</b> The evaluation pipeline, from near-duplicate hashing and group splitting to the external test, the audits and the illustrative simulation
</p>
<br>

</div>

<div align="center">

<p align="center">
<img src="FractureDetection/fig/layers_final.png" alt="Layers">
<br>
<b>Figure 2.</b> One detected fracture traced through the stem and the four stages of the network, ending on the Grad-CAM overlay. Stage labels are informal and the maps show mean absolute activation
</p>
<br>

</div>

Figure 2 is a visual guide, not evidence. Sections 7 and 9 test quantitatively where the model looks.

---

## 5. Internal Performance

The pooled out-of-fold AUC on FracAtlas is 0.923 (95% CI 0.910 to 0.935). This pools threshold-shifted scores. The raw pooled AUC is 0.911, and the mean of the five per-fold AUCs is 0.932. At the operating point, sensitivity is 0.808 (581 of 719 fractures detected) and specificity is 0.897.

Two hardware checks find no obvious shortcut inside FracAtlas. FracAtlas marks 99 images as containing metal implants, and 97 of them are fractured. Removing them moves the AUC to 0.915 (0.899 to 0.928). The hardware flag alone predicts fracture with an AUC of 0.567. Image-size class alone reaches 0.536.

A naive split that ignores near-duplicate groups gave an AUC of 0.931, against 0.923 for the controlled split. The intervals overlap and each split was trained once, so this comparison shows no measurable leakage inflation.

Performance varies with fracture size. This study defines a **small fracture** as a fractured image whose box area, relative to the image, lies in the smallest quartile. This gives 180 images. The model detects 76.1% of small fractures (69.3 to 82.2) and 88.9% of the largest quartile (83.9 to 93.2). Section 10 returns to this gap.

---

## 6. External Transfer

On GRAZPEDWRI-DX the five-fold ensemble reaches an AUC of 0.825 (0.816 to 0.833). Table 2 breaks this number down. The first row gives the closest internal comparison, the 1,281 FracAtlas hand images. The other rows use subsets defined by the GRAZPEDWRI-DX `cast`, `metal` and `initial_exam` fields.

<div align="center">

| Subset | AUC | Sensitivity | Specificity |
|---|---|---|---|
| FracAtlas hand, internal | 0.899 (0.876 to 0.920) | 0.866 | 0.742 |
| GRAZ, all images | 0.825 (0.816 to 0.833) | 0.776 | 0.749 |
| GRAZ, no cast and no metal | 0.744 (0.735 to 0.753) | 0.61 | 0.79 |
| GRAZ, initial exams only | 0.697 (0.687 to 0.707) | 0.540 | 0.790 |
| GRAZ, cast present (96% fractured) | 0.560 | 0.995 | 0.017 |

<b>Table 2.</b> Internal and external performance at the operating point fixed on FracAtlas validation folds. Intervals are 95% group bootstrap (FracAtlas groups for the first row, GRAZPEDWRI-DX patients for the rest)

</div>

Removing the 6,214 images with a cast or metal leaves 14,113 images, and the AUC falls to 0.744. Initial exams alone give 0.697, with sensitivity 0.540. Both subsets were chosen after the results were seen. Almost no initial exam carries a cast (2 of 10,861 images) or metal (17). The fall from 0.744 to 0.697 is therefore not a cast effect. First-visit fractures are simply harder for this model.

The problem lies in the ranking of images, not only in the threshold. Refitting the threshold on GRAZPEDWRI-DX to 90% specificity yields a sensitivity of only 0.673. This refit is excluded from the main analysis. The five fold models, scored individually, give external AUCs from 0.753 to 0.833. Their external specificity at their own cut-offs ranges from 0.356 to 0.785. A single trained model would therefore give an unstable external estimate. The five-fold ensemble's cut-off of zero was never checked on held-out external data.

The data show no device effect, though only a large one could be detected. The 112 Agfa images give an AUC of 0.774 (0.640 to 0.892), against 0.825 for Siemens images. Within the same patient, label and view, Agfa images score 0.40 logits lower than Siemens images (-1.32 to +0.48). Figure 3 shows the internal and external contrast as ROC curves.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/roc_internal_vs_external.png" alt="ROC curves" width="46%">
<br>
<b>Figure 3.</b> ROC curves for FracAtlas (internal, out of fold), GRAZPEDWRI-DX (external, all images) and GRAZPEDWRI-DX without cast or metal. Dots mark the operating point fixed in advance
</p>
<br>

</div>

In Figure 3 the external curve without casts or metal lies clearly below the full external curve. Casted and metal-containing images raise the headline number. Section 7 shows why.

---

## 7. The Cast Shortcut

### 7.1 A confounded first look

GRAZPEDWRI-DX contains 5,776 casted images, and 96% of them are fractured. The model flags 99.5% of the fractured casted wrists and 98.3% of the non-fractured ones. The model's own score separates cast from no cast with an AUC of 0.932. It separates fracture from no fracture with an AUC of 0.825.

A pooled comparison of casted and non-casted images is confounded. All 237 casted non-fractured images are follow-up exams. Cast and initial exam are almost perfectly separated. Non-cast follow-ups already score high. Figure 4a shows the mean score in each cell.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/cast_effect.png" alt="Cast effect">
<br>
<b>Figure 4.</b> (a) Mean ensemble score by label, exam type and cast. (b) Within-patient cast effect in fractured follow-up wrists, overall and by view, with 95% patient-bootstrap intervals. Zero is the decision threshold
</p>
<br>

</div>

Figure 4a shows three steps. At initial exams without a cast, fractured images score about 2.1 logits above non-fractured ones (+0.93 against -1.16). Among non-cast follow-ups the two labels score almost the same (+2.88 against +2.83, a gap of about 0.05). A cast then adds about 4 logits for both labels. Within follow-ups, the cast raises the mean by 3.84 logits for non-fractured images (2.71 to 5.13; 244 against 237 images). It raises the mean by 4.21 for fractured images (4.04 to 4.37). The model therefore responds to treatment status. Follow-up exams score high even without a cast.

### 7.2 Within-patient test

A within-patient design removes differences between patients. The test pairs follow-up images of the same patient, label and view, with and without a cast. In 946 patients with fractured wrists, a cast raises the score by 4.16 logits (3.99 to 4.31). The effect is +5.68 on frontal views and +2.61 on lateral views (Figure 4b). A patient fixed-effects regression gives a cast coefficient of +4.12 (3.97 to 4.27). The non-fractured within-patient contrast rests on only 10 patients (+6.65, 4.80 to 8.28). That estimate is too small a sample to rely on.

In 96% of pairs the casted exam came first. Time since injury is therefore partly mixed into the cast effect. The cast effect is still about twice the fracture effect at first presentation.

### 7.3 What the features encode

A linear probe asks which properties the model's features make readable. The probe uses the 768-dimensional pooled features of each fold model on 3,967 GRAZPEDWRI-DX images. It holds out whole patients. Figure 5 compares it with a probe on raw pixels.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/probe_auc.png" alt="Probe AUC">
<br>
<b>Figure 5.</b> Patient-held-out probe AUC for cast, follow-up status and fracture, on model features and on raw pixels, with 95% patient-bootstrap intervals
</p>
<br>

</div>

The features encode cast almost perfectly (AUC 0.998). They encode follow-up status at 0.960, and at 0.913 among non-casted images. Fracture reaches 0.907, and only 0.779 among casted images. Raw pixels already read cast at 0.935. A probe therefore shows that cast is readable. It does not show that the classifier uses it.

### 7.4 Grad-CAM on casted wrists

Grad-CAM [4] from the five fold models was averaged on 600 GRAZPEDWRI-DX images, half casted and half not. Each image was split by intensity into dark (air), mid (soft tissue) and bright (bone-like) regions. On casted wrists, the share of the top 10% of CAM mass on bright pixels falls from 0.356 to 0.226. The difference is -0.13 (-0.16 to -0.10), while the bright area stays about 19% in both groups. The intensity split cannot separate plaster from bone, so this is a correlate. Figure 6 shows random examples.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/cam_cast_contact_sheet.png" alt="Grad-CAM on casted wrists" width="70%">
<br>
<b>Figure 6.</b> Grad-CAM (mean of the five fold models) on random GRAZPEDWRI-DX wrists: casted without fracture (top), casted with fracture (middle) and non-casted without fracture (bottom). The cross marks the CAM peak
</p>
<br>

</div>

In Figure 6 every casted image is flagged, including all six without a fracture. On casted wrists the heat often runs along the cast shell and the edge of the limb. In 9 of the 12 casted examples the peak falls on a border, dark or mid region rather than bone. In the reference row the peaks also sit often at the border, sometimes on a side marker. Two reference images are flagged, and one of them carries a metal plate (score +11.4). The sheet illustrates the pattern; the 600-image contrast above measures it.

### 7.5 Perturbations

Perturbations test which image content drives the high casted scores. Gaussian blur with sigma 4 lowers casted scores by 5.6 logits, against 1.3 for non-casted images. Masking the outer 10% border lowers them by 2.0, against 0.7. Masking the brightest 5% of pixels does not remove the effect. Every flat mask raises scores, including a shifted control mask, so this test is limited by a mask artefact. Even after blur or border removal, 83% to 97% of casted images stay flagged. These results point to fine texture and peripheral content. They are suggestive, not causal.

### 7.6 Where the shortcut was learned

FracAtlas has no cast field, so one reader inspected FracAtlas images. The reader was an AI model, not a radiologist. Casts or splints appeared in 16 of 60 random fractured images and in 1 of 60 correctly rejected non-fractured images (p < 0.001). In FracAtlas, a cast is therefore close to a fracture label. Among the 60 highest-scoring false alarms, only 5 showed a cast (against 1 of 60, p = 0.21). Metal, mostly external objects such as bangles and watches, was more common among false alarms (14 against 3, p = 0.007). FracAtlas plausibly taught the cast cue. This study does not prove that the GRAZPEDWRI-DX behaviour comes from it.

---

## 8. View, Fracture Type, Age and Sex

On initial exams the model is weakest on frontal views. Frontal views give an AUC of 0.61 and a sensitivity of 31% (5,370 images). Lateral views give 0.79 and 73% (5,439 images). Combining the two views of one study did not help. On 4,951 first-visit studies with both views, the lateral view alone gives 0.785 (0.771 to 0.797) and the frontal view alone 0.580. The maximum of the two scores gives 0.775, and their mean gives 0.743.

By AO fracture class, first-visit sensitivity is 51% for the most common class, 23r-M (3,173 images). It is 67% for 23-M, 49% for 23r-E and 79% for 22r-D (62 images). Counts are images; only classes with at least 30 patients are shown.

Age and sex were tested on no-cast initial exams, where the cast and follow-up effects are absent. Figure 7 shows the result.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/age_sex.png" alt="Age and sex">
<br>
<b>Figure 7.</b> AUC (a) and sensitivity at the fixed threshold (b) on no-cast initial exams, by age band and sex, with 95% patient-bootstrap intervals
</p>
<br>

</div>

Every subgroup in Figure 7 lies between AUC 0.65 and 0.72. Patients aged 16 and over are weakest, with AUC 0.645 and sensitivity 0.45. Sensitivity falls steadily with age, from 0.58 under 6 years to 0.45 at 16 and over. Males reach an AUC of 0.717 and females 0.675, and their intervals do not overlap.

---

## 9. Where the Model Looks on Fractures

This audit returns to FracAtlas, where every fracture has a box. For each fractured image, Grad-CAM is computed on the last stage of the fold model that scored it. The audit records whether the heatmap peak falls inside the box, enlarged by 16 pixels on each side. The peak falls inside for 63.7% (59.6 to 67.8) of the 581 detected fractures. It falls inside for only 29.7% (22.3 to 38.1) of the 138 missed ones. A uniformly placed peak would land in the enlarged box about 3% of the time. This chance level is generous to the model.

An occlusion test checks the same question by intervention. Covering the fracture with the image's median grey lowers the score by 2.42 logits on average (2.13 to 2.72). Covering up to five same-sized control patches on bright areas away from the fracture raises the score by about 0.01 logits. Covering the fracture lowers the score more than the control patches in 77% of cases (73 to 80%). After the fracture is covered, 86% of these images are still flagged. Figures 8 to 10 show individual cases.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/grid_detected.png" alt="Detected">
<br>
<b>Figure 8.</b> Correctly detected fractures, randomly sampled, with the annotated box in green and the Grad-CAM overlay below
</p>
<br>

</div>

<div align="center">

<p align="center">
<img src="FractureDetection/fig/grid_missed.png" alt="Missed">
<br>
<b>Figure 9.</b> Missed fractures (fractured images scored below the operating point), randomly sampled
</p>
<br>

</div>

<div align="center">

<p align="center">
<img src="FractureDetection/fig/grid_false_alarms.png" alt="False alarms">
<br>
<b>Figure 10.</b> The six highest-scoring non-fractured images, which are the model's most confident false alarms
</p>
<br>

</div>

In Figure 8 the heatmaps of detected fractures tend to sit on the box. In Figure 9 the heatmaps of missed fractures often do not. Figure 10 shows the confident false alarms that Section 7.6 examined. On average the model uses the fracture region, but the fracture alone does not carry the decision.

---

## 10. Small Fractures

Section 5 showed that small fractures are detected less often (76.1% against 88.9%). This section asks whether that gap can be closed and whether a realistic gain can be measured. The primary metric is sensitivity on the 180 small fractures at the fixed operating point, with the same folds in every run.

Resolution was ruled out first. Reducing images to 128 pixels and restoring them to 512 lowers the AUC from 0.923 to 0.850, mostly through false alarms. Small-fracture sensitivity does not fall (0.76 native, 0.77 to 0.80 degraded). Upscaling the input to 640 or 768 pixels adds nothing.

Three interventions were tested. The first is an auxiliary head that predicts a coarse fracture-location map during training only. It is supervised with boxes on 100%, 25% or 10% of fractured training images. The second is a finer version of that head. The third oversamples small fractures three-fold. Each box run has a matching control that trains the same head without boxes. The unmodified baseline was trained with seeds 42 and 43 on the same folds.

A decision rule was written down before the fine head, its control and the oversampling run were trained. An intervention counts only if its paired 95% interval lies above zero against both baseline seeds. It must also lose no more than 0.01 AUC or 0.03 specificity. The three box runs and the first control had already been seen, so they are exploratory. The rule was tightened once, after the seed-43 baseline appeared. Table 3 compares every run with the seed-43 baseline, the stronger of the two.

<div align="center">

| Run | AUC | Small-fracture sensitivity | Difference from seed-43 baseline |
|---|---|---|---|
| Baseline, seed 42 | 0.923 | 0.761 (0.693 to 0.822) | |
| Baseline, seed 43 | 0.932 | 0.794 (0.735 to 0.854) | |
| Auxiliary head, no boxes (control) | 0.935 | 0.806 | +0.011 (-0.033 to +0.052) |
| Boxes on 100% of fractured images | 0.936 | 0.833 (0.773 to 0.885) | +0.039 (-0.005 to +0.078) |
| Boxes on 25% | 0.932 | 0.800 | +0.006 (-0.043 to +0.051) |
| Boxes on 10% | 0.938 | 0.828 (0.769 to 0.886) | +0.033 (-0.018 to +0.078) |
| Fine head, boxes on 100% | 0.939 | 0.828 (0.770 to 0.881) | +0.033 (-0.013 to +0.080) |
| Fine head, no boxes (control) | 0.936 | 0.794 | +0.000 (-0.045 to +0.045) |
| Oversample small x3 | 0.932 | 0.761 (0.688 to 0.826) | -0.033 (-0.086 to +0.015) |

<b>Table 3.</b> Out-of-fold AUC and sensitivity on the 180 small fractures at the fixed operating point. Differences are paired, with 95% group-bootstrap intervals

</div>

No run passes the rule. No interval clears zero against seed 43, and no box run beats its own no-box control. The fine head with boxes beats its control by +0.033 (-0.010 to +0.076). The direction is consistent across comparisons. For the six auxiliary-head and box runs, gains range from 0.000 to 0.039 against seed 43. Against seed 42 the same runs gain 0.033 to 0.072. Against the mean of the two baselines they gain 0.017 to 0.056. That last comparison was chosen after the results were seen. Oversampling gains -0.033, 0.000 and -0.017 under the three comparisons. Figure 11 places these estimates against the two baseline seeds.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/small_fracture_runs.png" alt="Small fracture runs">
<br>
<b>Figure 11.</b> Small-fracture sensitivity by run, with 95% intervals. The shaded band spans the two baseline seeds
</p>
<br>

</div>

The band in Figure 11 is the main result of this section. The two baseline seeds differ by 0.033 with no change to the method, and 18 of the 180 images changed outcome. Seed variation is therefore as large as the gains under test. The paired 95% half-width on 180 images is about 0.045. Only differences of about 0.065 or more are detectable in principle with this design. The rule stopped three single-seed wins against seed 42 from being reported as findings. Figures 12 and 13 show the hardest cases.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/grid_small.png" alt="Small fracture grid">
<br>
<b>Figure 12.</b> Small fractures, three detected (left) and three missed (right), randomly sampled
</p>
<br>

</div>

<div align="center">

<p align="center">
<img src="FractureDetection/fig/gradcam_small.gif" alt="Small fracture animation" width="85%">
<br>
<b>Figure 13.</b> A detected and a missed small fracture. The annotated box appears first and the Grad-CAM heatmap follows. In the missed case the heatmap peak lies away from the box
</p>
<br>

</div>

---

## 11. Illustrative ED Simulation

A discrete-event simulation translates the two measured operating points into missed fractures and workload. It models one typical English major (Type 1) emergency department (ED) for a year, averaged over 10 simulated years. Only the national volume is sourced, from NHS England data (1.503 million Type 1 attendances in September 2026). The other inputs are assumptions. They include 120 departments, about 49 limb X-rays a day and a fracture prevalence of 20%. They also include a **usual miss rate** of 5% by the ED clinician before the radiology report. An AI alert on a missed fracture leads to correction with probability 0.8.

As an **alert**, a flagged image prompts a re-read while the patient is still in the ED. As **triage**, flagged images move to the front of the reporting queue. The simulation applies a missed-fracture sensitivity of 0.761 for the internal point. This is the small-fracture sensitivity, used as a proxy for subtle misses. For the external point it applies 0.509, which is 0.54 x 0.761 / 0.808.

Under these assumptions the ED misses about 182 fractures a year. The internal point (sensitivity 0.81, specificity 0.90) catches 109 of them. It costs about 12 extra re-reads a day, of which 4 are false alarms. The external first-visit point (sensitivity 0.54, specificity 0.79) catches 71. It costs 13.5 re-reads a day, of which 8.2 are false alarms. Triage helps most when the reporter is busy. At 95% reporter load, the internal point lowers undetected fracture-hours from 1,760 to 1,110 a year.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/ed_simulation.png" alt="ED simulation">
<br>
<b>Figure 14.</b> Illustrative ED simulation at a 5% usual miss rate, comparing the internal and external operating points. A comparison of assumptions, not a prediction
</p>
<br>

</div>

Figure 14 shows that planning around the internal figure overstates caught misses and understates false alarms. The simulation code can calibrate arrivals and reporting times from a site's timestamps. It becomes a site model only after calibration and validation on real local data.

A companion break-even analysis computes the highest cost-neutral price per scan. It uses a 5 x 5 grid of unsourced values for the cost of a missed fracture and the usual miss rate. At a 10% miss rate and a cost of 1,000 per miss, the price is 3.2 at the external point and 12.6 at the internal point. The external price is negative in 11 of 25 cells, against 5 for the internal point.

<div align="center">

<p align="center">
<img src="FractureDetection/fig/breakeven.png" alt="Break-even">
<br>
<b>Figure 15.</b> Illustrative break-even grid. Costs and usual miss rates are axes, and none of the values are sourced
</p>
<br>

</div>

Figure 15 leads to the same conclusion as Figure 14. External performance, not internal performance, decides whether a fracture AI is viable.

---

## 12. Discussion

One model, measured four ways, gives answers that differ by more than their confidence intervals. The AUC is 0.923 on FracAtlas and 0.825 on GRAZPEDWRI-DX. It is 0.744 without casts or metal and 0.697 on first-visit exams. Part of this drop is a task change, from mixed adult anatomy to paediatric wrists. A random split of the source data would reveal none of it.

The largest single effect is treatment status. Inside the same patient, a cast raises the score by about 4 logits. At first presentation a fracture raises it by about 2 logits. Follow-up exams without a cast also score high, whatever their label. The features encode cast almost perfectly, Grad-CAM moves off bone on casted wrists, and blur removes much of the casted score. Together these findings fit a cast and treatment cue. None of them alone proves the mechanism.

This shortcut has a practical consequence. At first presentation no cast exists, and the model's performance there is the weakest. External results should therefore be reported with and without treatment artefacts. The first-visit figure (0.697) is the fairer headline for a triage use case.

The fracture-level audits on FracAtlas support the model on average. Heatmaps and occlusion agree that it often uses the fracture region. Most images stay flagged after the fracture is covered, so the fracture is not the whole story.

The small-fracture experiments are a finding about measurement. With two seeds and 180 images, a gain of a few points cannot be separated from seed variation. Published comparisons of similar size that report one training run deserve the same caution.

---

## 13. Contributions

1. **A fully open, leak-controlled evaluation pipeline.** Near-duplicate radiographs are grouped before splitting, and every FracAtlas image is scored by a model that never saw it. The threshold is fixed on validation data and never refitted, including on the external set.

2. **An audit of what the detector relies on.** Within-patient tests, feature probes, Grad-CAM and perturbations expose a cast and treatment shortcut. This shortcut moves the score more than a fracture does at first presentation. The audit also exposes collapse on first-visit exams and weak performance on small fractures and older patients.

3. **A measured account of what the data can resolve.** The pipeline releases a pre-written decision rule, a seed-variance check, a resolution test and an illustrative ED simulation as code. Any other fracture model can be put through the same questions.

---

## 14. Limitations and Future Work

This is a retrospective study on public data, and it makes no clinical claim. Every setting was trained once, with one extra seed for the baseline only. Run-to-run variance is therefore not estimated, and single intervals understate the uncertainty.

FracAtlas has no patient identifiers. Leakage between images of one patient that the hash grouping missed remains possible. The naive-versus-controlled comparison cannot measure its size. FracAtlas mixes body parts and contains only 381 fractured hand images.

GRAZPEDWRI-DX is paediatric and wrist-only. Its 112 Agfa images can rule out only very large device effects. Images with a blank `initial_exam` field were treated as follow-ups. The provenance of the `fracture_visible` label was not checked. The licence terms of both datasets have not been verified and should be checked before any redistribution.

The cast, view, fracture-type, age, sex and device analyses are post hoc. So are the 0.744 and 0.697 subsets. The five-fold ensemble's cut-off of zero was never checked on held-out external data. The cast effect remains partly mixed with time since injury, because the casted exam came first in 96% of pairs. The non-fractured within-patient contrast rests on 10 patients.

The probe shows that information is readable, not that the classifier uses it. The Grad-CAM intensity split cannot separate plaster from bone. Perturbations push images off the training distribution, and the bright-pixel mask is limited by a mask artefact. The occlusion control patch is chosen by brightness and may not lie on bone. The FracAtlas visual audit used one reader, an AI model and not a radiologist, on 512-pixel images, with no second reader. The reader knew that the random set was fractured, which may raise its cast count.

The small-fracture decision rule was self-timestamped, not externally registered, and was tightened once. The ED simulation rests on assumptions, and the break-even values are unsourced. Neither is evidence of clinical benefit.

The most useful extensions follow from these limits. A same-wrist counterfactual test, with and without a cast, would establish the shortcut causally. Training with treatment-balanced data, or with casted images excluded, would show whether first-visit accuracy improves. The small-fracture test needs three to five seeds per arm. Training at higher resolution, on paired views or with GRAZPEDWRI-DX patients was not tried. Prospective multi-site evaluation remains the standard for any such system.

---

## 15. Repository Layout and Reproduction

All code lives in `FractureDetection/` in this repository. The `fracai/` package holds the library code: data, split, training, metrics, audit, Grad-CAM, break-even and simulation. The `scripts/` folder has one subfolder each for data preparation, training, audits, analysis and simulation. The `jobs/` folder holds the resumable shell runners. The `docs/` folder holds the analysis plan, limitations, data licence notes and the paper draft. The test suite has 28 tests, run with `pytest`, and `ruff` passes.

```
cd FractureDetection
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Run every script from the `FractureDetection` folder with `PYTHONPATH=.`. The order is the data scripts, then `bash jobs/run_training.sh`, then the box and audit scripts. The control and small-fracture runs, the analysis scripts and the simulation follow. The shortcut audit runs with `bash jobs/run_audits.sh`. It calls `scripts/analysis/a1_cast_within_patient.py`, `a2_age_sex.py` and `a3_device.py` on the saved external scores. It then calls `scripts/audit/b1_feature_probe.py`, `b2_cam_cast.py` and `b3_perturb.py` on the saved fold models, and `scripts/analysis/audit_figures.py` for the figures. The FracAtlas visual audit uses `scripts/analysis/cast_contact_sheets.py`, `cast_contact_sheets_fxrand.py` and `cast_annotation_summary.py`.

Datasets and model weights are not included and must be downloaded from their original sources. Re-running the audits from the saved score files in a clean copy reproduced the reported numbers. A full retraining from scratch has not been repeated.

---

## 16. References

1. Abedeen et al. (2023). FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs. Scientific Data 10.
2. Nagy et al. (2022). A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for machine learning. Scientific Data 9.
3. Liu et al. (2022). A ConvNet for the 2020s. CVPR 2022.
4. Selvaraju et al. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. International Journal of Computer Vision 128.
5. Geirhos et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence 2.
6. Zech et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLOS Medicine 15.
7. Ju and Cai (2023). Fracture detection in pediatric wrist trauma X-ray images using YOLOv8 algorithm. Scientific Reports 13.
8. Zech et al. (2023). Detecting pediatric wrist fractures using deep-learning-based object detection. Pediatric Radiology 53.
9. Yoon et al. (2021). Development and Validation of a Deep Learning Model Using Convolutional Neural Networks to Identify Scaphoid Fractures in Radiographs. JAMA Network Open 4.

---

## Author

Navin Bondade
Researcher, University College London
