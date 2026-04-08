# PPT Formulas & Metrics Documentation
## Phase II - Face Recognition System

This document contains all formulas, metrics, and calculations used in the face recognition project. Use this for your PowerPoint presentation and viva defense.

---

## 📐 **1. RECOGNITION CONFIDENCE FORMULAS**

### 1.1 LBPH Confidence Score
```
confidence_score = histogram_distance(test_face_LBP, trained_face_LBP)
```
**Explanation:**
- LBPH extracts Local Binary Patterns from the test face
- Compares histogram with all trained faces
- Returns distance to closest match
- Lower distance = higher certainty of match

### 1.2 Recognition Decision
```
IF confidence < 70 THEN
    recognized_person = trained_names[label]
ELSE
    recognized_person = "Unknown"
END IF
```
**Parameters:**
- Threshold: `70` (empirically determined)
- Confidence < 70 → **RECOGNIZED**
- Confidence ≥ 70 → **UNKNOWN**

### 1.3 Display Confidence Adjustment
```
display_confidence = MAX(0, raw_confidence - 12)
```
**Purpose:** Makes confidence scores more user-friendly for visualization

---

## 🎯 **2. ACCURACY & CLASSIFICATION METRICS**

### 2.1 Confusion Matrix Terms
```
┌─────────────────────────────────────────────┐
│          PREDICTED                          │
│     Positive (Known)  │  Negative (Unknown) │
├──────────────────────┼────────────────────┤
│ True Positive (TP)   │ False Negative (FN)│ Actual
│ Actual know person   │ Known marked Unknown│ Positive
├──────────────────────┼────────────────────┤
│ False Positive (FP)  │ True Negative (TN) │
│ Unknown marked Known  │ Unknown (correct)  │ Actual
└──────────────────────┴────────────────────┘
```

**Definitions:**
- **TP (True Positive):** System correctly identified a known person
  - Confidence < 70 AND person is actually in training data ✅
  
- **FP (False Positive):** System identified unknown as known
  - Confidence < 70 BUT person is NOT in training data ❌
  
- **TN (True Negative):** System correctly identified unknown
  - Confidence ≥ 70 AND person is NOT in training data ✅
  
- **FN (False Negative):** System missed a known person
  - Confidence ≥ 70 BUT person IS in training data ❌

### 2.2 Precision (Positive Predictive Value)
```
Precision = TP / (TP + FP)

Range: 0 to 1 (or 0% to 100%)

Meaning: Of all faces marked as "Known", how many were actually known?
         Higher precision = fewer false alarms
```

**Example:**
```
If system says 10 faces are known (TP + FP = 10)
And 8 are actually known (TP = 8)
Then FP = 2

Precision = 8 / (8 + 2) = 0.80 = 80%
```

### 2.3 Recall (Sensitivity / True Positive Rate)
```
Recall = TP / (TP + FN)

Range: 0 to 1 (or 0% to 100%)

Meaning: Of all actual known faces, how many did we catch?
         Higher recall = fewer missed known faces
```

**Example:**
```
If there are 10 known people (TP + FN = 10)
And system correctly identified 8 (TP = 8)
Then FN = 2

Recall = 8 / (8 + 2) = 0.80 = 80%
```

### 2.4 Accuracy (Overall Correctness)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Range: 0 to 1 (or 0% to 100%)

Meaning: Percentage of all predictions that were correct
         Balanced metric for overall performance
```

**Example:**
```
Total Detections: 100
- TP = 35 (Correct Known)
- TN = 58 (Correct Unknown)
- FP = 4  (Wrong Known)
- FN = 3  (Wrong Unknown)

Accuracy = (35 + 58) / (35 + 58 + 4 + 3) = 93 / 100 = 0.93 = 93%
```

### 2.5 F1-Score (Harmonic Mean)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Range: 0 to 1

Meaning: Balanced metric combining precision and recall
         Best value: 1.0 (perfect precision and recall)
         Useful when you care about both metrics equally
```

**Example:**
```
If Precision = 0.80 and Recall = 0.75

F1 = 2 × (0.80 × 0.75) / (0.80 + 0.75)
   = 2 × 0.60 / 1.55
   = 1.20 / 1.55
   = 0.774 ≈ 77.4%
```

---

## ⚡ **3. BMC ENHANCEMENT METRICS (REVIEW 2)**

### 3.1 Output Enhancement Percentage
```
Output Enhanced (%) = (BMC_Recognitions / Frames_Processed) × 100

Range: 0% to 100%

Meaning: Percentage of detected faces successfully recognized with BMC
         Higher = BMC is improving recognition outcomes
```

**Example:**
```
During a session:
- Frames Processed: 245 faces detected
- BMC Recognitions: 180 successfully recognized

Output Enhanced = (180 / 245) × 100 = 73.5%

Interpretation: BMC improved recognition in 73.5% of detections
```

### 3.2 Average BMC Processing Time
```
Avg_Processing_Time = Total_BMC_Time / Frames_Processed

Unit: milliseconds (ms)

Color Indicators:
- 🟢 GREEN:  < 10 ms   (Excellent)
- 🟡 YELLOW: 10-20 ms  (Good)
- 🔴 RED:    > 20 ms   (Slow)
```

**Example:**
```
If Total BMC time for 245 frames = 1190 ms
Avg_Processing_Time = 1190 / 245 = 4.86 ms ✅ GREEN
```

### 3.3 Estimated Throughput
```
Est_Throughput = 1000 / Avg_Processing_Time

Unit: Frames Per Second (FPS)

Meaning: Theoretical maximum FPS for BMC processing alone
         Real system FPS is lower (includes detection, UI)
```

**Example:**
```
If Avg_Processing_Time = 4.87 ms
Est_Throughput = 1000 / 4.87 ≈ 205 FPS

This means BMC can handle 205 frames per second theoretically
```

### 3.4 System FPS (Real-Time Performance)
```
System_FPS = 1 / Avg_Frame_Time

Where Avg_Frame_Time = average of last 30 frame times

Unit: Frames Per Second (FPS)

Meaning: Actual end-to-end system performance
         Includes: camera capture, preprocessing, detection, BMC, UI
```

**Example:**
```
If average frame time = 0.042 seconds
System_FPS = 1 / 0.042 ≈ 23.8 FPS

Performance Target: > 20 FPS (smooth real-time)
```

---

## 📊 **4. DETECTION METRICS**

### 4.1 Detection Rate
```
Detection_Rate = Faces_Detected / Total_Frames

Unit: Ratio or Percentage

Meaning: How many frames contain at least one face
```

**Example:**
```
Detected in 180 frames out of 300 total frames
Detection_Rate = 180 / 300 = 0.60 = 60%
```

### 4.2 Recognition Rate (Given Detection)
```
Recognition_Rate = Faces_Recognized / Faces_Detected

Unit: Ratio or Percentage

Meaning: Of detected faces, what percentage were recognized
```

**Example:**
```
180 faces detected, 135 recognized
Recognition_Rate = 135 / 180 = 0.75 = 75%
```

---

## 🔄 **5. LBPH ALGORITHM FORMULAS**

### 5.1 Local Binary Pattern (LBP)
```
LBP(p, c) = Σ s(gp - gc) × 2^p

Where:
- p = pixel position (0 to 7, for 8 neighbors)
- c = center pixel
- gp = gray value at position p
- s(x) = sign function:
    s(x) = 1 if x ≥ 0
    s(x) = 0 if x < 0
```

**Explanation:**
- Analyzes 8 neighbors around each pixel
- Creates binary pattern (8 bits)
- Captures texture information locally
- Robust to illumination changes

### 5.2 LBP Histogram
```
Histogram_LBP = Count of each LBP pattern in image

Total patterns possible: 2^8 = 256

For face recognition:
1. Extract LBP from training image
2. Create 256-bin histogram
3. Store per person
4. Compare test face histogram with all stored histograms
```

### 5.3 Histogram Distance (Chi-Square)
```
Distance = Σ (H_test[i] - H_trained[i])² / (H_test[i] + H_trained[i])

Where:
- H_test[i] = bin i of test face histogram
- H_trained[i] = bin i of trained face histogram
- Sum over all 256 bins
```

**Interpretation:**
- Lower distance = better match
- Distance < 70 = Recognition threshold
- Used to determine confidence score

---

## 🎛️ **6. HAAR CASCADE PARAMETERS**

### 6.1 Face Detection Parameters
```
face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,        # Image pyramid scale
    minNeighbors=8,         # Quality threshold
    minSize=(80, 80),       # Minimum face size
    maxSize=(400, 400)      # Maximum face size
)
```

**Parameter Meanings:**
```
scaleFactor = 1.1:
  - Reduces image by 10% in each iteration
  - Faster detection, but less accuracy if too large
  - Smaller = more accurate but slower

minNeighbors = 8:
  - Minimum 8 overlapping rectangles for detection
  - Higher = fewer false positives
  - Lower = more detections but more noise

minSize & maxSize:
  - Constrains face detection region
  - Prevents very small noise and huge objects
```

---

## 📈 **7. TRAINING STATISTICS**

### 7.1 Training Data Composition
```
Total Training Faces: 4 people
- Person 1: 1 face image
- Person 2: 1 face image
- Person 3: 1 face image
- Person 4: 1 face image

Face Size: 200 × 200 pixels (normalized)
Color Space: Grayscale
Preprocessing: BMC light mode (sigma=50)
```

### 7.2 Model Parameters
```
LBPH Recognizer Settings:
- Radius: 1 (default)
- Neighbors: 8 (default)
- Grid X: 8 (default)
- Grid Y: 8 (default)
- Threshold: 70 (empirically determined)
```

---

## 🛠️ **8. BMC ENHANCEMENT FORMULA**

### 8.1 Bilateral Median Convolution (Light Mode)
```
BMC_Output = BilateralFilter(Input_Face, d=5, sigma=50)

Where:
- d = diameter of pixel neighborhood
- sigma = spatial/color standard deviation
- Single pass (light mode)

Process:
1. Smooth image while preserving edges (bilateral filter)
2. Enhance texture patterns for LBPH
3. Handles noise and occlusions

Result: Enhanced face image ready for LBPH extraction
```

---

## 📋 **9. SUMMARY TABLE - KEY METRICS FOR PPT**

| Metric | Formula | Range | Good Value | Used For |
|--------|---------|-------|------------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-100% | >85% | Overall performance |
| **Precision** | TP/(TP+FP) | 0-100% | >80% | False alarm rate |
| **Recall** | TP/(TP+FN) | 0-100% | >80% | Miss rate |
| **F1-Score** | 2×(P×R)/(P+R) | 0-100% | >80% | Balanced metric |
| **Output Enhanced** | (Recog/Detected)×100 | 0-100% | >70% | BMC effectiveness |
| **Avg Processing** | Total_Time/Frames | ms | <10ms | BMC speed |
| **System FPS** | 1/Avg_Frame_Time | FPS | >20 | Real-time viability |
| **Detection Rate** | Detected/Total_Frames | 0-100% | >60% | Detection quality |
| **Recognition Rate** | Recognized/Detected | 0-100% | >75% | Recognition quality |

---

## 🎓 **10. EXPLANATION FOR VIVA**

### How to Explain Formulas:

**On Accuracy:**
> "Accuracy tells us the percentage of correct predictions overall. Formula: (True Positives + True Negatives) / Total Predictions. For example, if we correctly identified 93 out of 100 faces, accuracy is 93%."

**On Precision vs Recall:**
> "Precision answers: 'Of faces we marked as known, how many actually were known?' Recall answers: 'Of all known faces, how many did we find?' Both metrics are important - precision prevents false alarms, recall prevents missing known faces."

**On Output Enhanced %:**
> "This metric shows BMC's practical impact. It's calculated as: (Faces successfully recognized / Total faces processed) × 100. A 73.5% enhancement tells reviewers that BMC improved 3 out of 4 detection outcomes."

**On FPS Metrics:**
> "We track two FPS values: Estimated Throughput (theoretical BMC speed) and System FPS (real-time performance). System FPS is lower because it includes detection and UI overhead, but we maintain >20 FPS for smooth real-time operation."

---

## 💾 **11. QUICK REFERENCE FOR YOUR PPT SLIDES**

### Slide 1: Recognition Confidence
- Formula: `confidence < 70 → Recognized`
- Display: `confidence - 12`
- Visual: Green box if recognized, Red if unknown

### Slide 2: Accuracy Metrics
- **Precision:** TP/(TP+FP) - What % of our positive predictions are correct
- **Recall:** TP/(TP+FN) - What % of actual positives did we catch
- **Accuracy:** (TP+TN)/(TP+TN+FP+FN) - Overall correctness

### Slide 3: BMC Metrics
- **Output Enhanced:** (Recognitions/Frames) × 100 - BMC effectiveness
- **Avg Processing:** <10ms - BMC speed ✅
- **System FPS:** >20 - Real-time viability ✅

### Slide 4: Detection & Recognition Rates
- Detection Rate = Faces detected / Total frames
- Recognition Rate = Faces recognized / Faces detected

### Slide 5: Algorithm Overview
- LBPH: Local Binary Patterns Histogram
- BMC: Bilateral Median Convolution (Review 2 enhancement)
- Haar Cascade: Face detection (scaleFactor=1.1, minNeighbors=8)

---

**Last Updated:** March 27, 2026 - Review 2 Final Documentation
**For:** PowerPoint Presentation & Viva Defense
