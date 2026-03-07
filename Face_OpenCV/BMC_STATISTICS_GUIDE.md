# BMC Statistics Feature - Quick Guide

## 📊 How to Use BMC Statistics

### Step 1: Launch the System
```bash
python Face_App.py
```

### Step 2: Toggle Statistics ON
Press **`T`** key during recognition to show the BMC statistics panel

### Step 3: View Real-Time Metrics

The statistics panel appears in the **top-right corner** and shows:

```
┌──────────────────────────────────────┐
│       BMC STATISTICS                 │
│                                      │
│  Frames Processed: 245               │
│  Avg Processing: 4.87 ms  [GREEN]   │
│  BMC Recognitions: 18                │
│  Status: ACTIVE (Light Mode)         │
│  Est. Throughput: ~205 FPS           │
└──────────────────────────────────────┘
```

### Step 4: Toggle Statistics OFF
Press **`T`** again to hide the panel

---

## 🎨 Color Indicators

### Processing Time Colors:
- **🟢 GREEN** (< 10ms): Excellent performance
- **🟡 YELLOW** (10-20ms): Good performance  
- **🔴 RED** (> 20ms): May need optimization

### Panel Features:
- **Animated glowing border** (green pulse)
- **Semi-transparent background**
- **Real-time updates** every frame
- **Pulsing title** for visual appeal

---

## 📈 Understanding the Metrics

### 1. **Frames Processed**
- Total number of face detections processed with BMC
- Increases each time a face is detected
- Reset with `R` key

### 2. **Avg Processing**
- Average time to apply BMC to one face
- Measured in milliseconds (ms)
- Lower is better (target: < 10ms)

### 3. **BMC Recognitions**
- Number of successful recognitions with BMC active
- Confidence score < 70 (match found)
- Shows BMC effectiveness

### 4. **Status**
- Confirms BMC is active
- Shows current mode (Light/Medium/Heavy)
- Light mode = Best compatibility

### 5. **Est. Throughput**
- Calculated from avg processing time
- Shows theoretical max FPS for BMC only
- Real system FPS is lower (includes camera, detection, UI)

---

## 🎮 Full Control Reference

| Key | Action | Effect |
|-----|--------|--------|
| **T** | Toggle Stats | Show/Hide BMC statistics panel |
| **Q** | Quit | Exit system with session summary |
| **F** | Fullscreen | Toggle fullscreen mode |
| **P** | Pause | Pause/Resume recognition |
| **R** | Reset | Clear all statistics including BMC |

---

## 💡 Tips for Best Results

1. **Monitor Performance:**
   - Check Avg Processing time
   - Green (< 10ms) = Optimal
   - If red (> 20ms), system may be overloaded

2. **Compare Recognition:**
   - Watch BMC Recognitions counter
   - Higher count = Better recognition with BMC
   - Compare with total detected faces

3. **Use During Debugging:**
   - Toggle ON when testing different people
   - Monitor if BMC helps or hurts recognition
   - Check processing overhead

4. **Reset for Clean Data:**
   - Press `R` to reset all counters
   - Start fresh testing session
   - Get accurate averages

---

## 🔧 Technical Details

### BMC Processing Pipeline (Light Mode):
```
Input Face (200x200 grayscale)
    ↓
Bilateral Filter (d=5, sigma=50)
    ↓
Output Face (enhanced features)
```

### Statistics Collection:
```python
# Timing measurement
bmc_start = time.time()
face = bmc_processor.fast_bmc(face, strength='light')
bmc_time = (time.time() - bmc_start) * 1000  # Convert to ms

# Update statistics
bmc_stats['total_processed'] += 1
bmc_stats['total_time'] += bmc_time
bmc_stats['avg_time_ms'] = bmc_stats['total_time'] / bmc_stats['total_processed']
```

---

## 📸 Screenshot Reference

### Panel Layout:
```
Position: Top-Right (20px margin)
Size: 350x180 pixels
Background: Semi-transparent dark (opacity: 85%)
Border: 3px animated green glow
Font: OpenCV Hershey Simplex

Title: "BMC STATISTICS" (0.7, bold, pulsing green)
Metrics: (0.5, normal, color-coded)
```

### Visual Elements:
- Pulsing border animation (10 FPS pulse)
- Title pulse (8.3 FPS pulse)
- Status indicator (5 FPS pulse)
- Color-coded performance metrics

---

## 🐛 Troubleshooting

### Panel Not Appearing?
- Check if BMC is enabled (should see "BMC Enhancement Module Loaded")
- Ensure you're pressing `T` (not Shift+T)
- Look at top-right corner (may be off-screen in windowed mode)

### Statistics Not Updating?
- Ensure faces are being detected (green/red boxes visible)
- BMC only processes detected faces
- If paused (`P` key), statistics won't update

### Performance Issues?
- Red processing time (> 20ms) indicates overload
- Close other applications using camera
- Consider reducing camera resolution in config.json

---

## ✅ Quick Test

1. Launch system: `python Face_App.py`
2. Press `T` - Panel should appear
3. Stand in front of camera - See counters increase
4. Press `R` - Counters reset to 0
5. Press `T` - Panel disappears

**Expected Result:** Avg processing time should be green (< 10ms) on modern CPUs

---

*Last Updated: March 7, 2026 | Review 2 - BMC Statistics Feature*
