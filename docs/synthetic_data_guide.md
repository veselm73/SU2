# Synthetic Data Generator - Quick Reference
# ===========================================

## Current Optimized Parameters

### Cell Generation
- **Cell Count:** 8-24 per patch (avg ~16)
- **Cell Radius:** 4.0 pixels
- **Cell Type:** Filled Gaussian (not rings)
- **Patch Size:** 128x128 pixels

### Intensity & Contrast
- **Foreground Contrast:** 0.4 - 1.2
- **Background Contrast:** 0.2 - 0.5
- **Foreground Intensity:** 400 + contrast×800
- **Background Intensity:** 100 + contrast×150
- **Normalization:** Min-max (preserves brightness)

### SIM Parameters
- **NA:** 1.49
- **Wavelength:** 512 nm
- **Pixel Size:** 0.07 μm
- **Wiener Parameter:** 0.1

## Performance Metrics

Comparison with validation data:
- **Mean Intensity Difference:** ~17% ✓
- **Std Deviation Difference:** ~15% ✓
- **Overall Match:** GOOD

## Key Files Modified

1. **modules/dataset.py**
   - Changed cell generation to filled Gaussian structures
   - Updated default contrast ranges

2. **modules/simulation.py**
   - Increased foreground/background intensity values
   - Changed normalization from z-score to min-max

3. **modules/config.py**
   - Added RADIUS parameter
   - Maintains MIN_CELLS, MAX_CELLS, PATCH_SIZE, SIM_CONFIG

## Quick Adjustments

**To increase brightness:**
```python
# In modules/dataset.py
contrast_fg_range=(0.5, 1.5)  # Increase upper bound
```

**To change cell density:**
```yaml
# In config.yaml
MIN_CELLS: 10
MAX_CELLS: 30
```

**To adjust cell size:**
```yaml
# In config.yaml
RADIUS: 5.0  # Larger cells
```

**To add more texture variation:**
```python
# In modules/simulation.py
self.perlin = PerlinNoise(self.patch_size, 2)  # Increase from 1 to 2
```
