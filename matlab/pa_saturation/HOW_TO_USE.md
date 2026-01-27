# FTN Neural Network Detection with Hardware Impairments - User Guide

## Overview

This package provides MATLAB implementations for **Faster-than-Nyquist (FTN) signaling** with **neural network-based detection** in the presence of **hardware impairments** (primarily Power Amplifier saturation).

### Key Features

1. **Hardware Impairments**:
   - Power Amplifier (PA) saturation (3 models: Rapp, Saleh, Soft Limiter)
   - IQ imbalance (TX/RX)
   - Phase noise
   - Carrier frequency offset (CFO)

2. **Advanced Detection Methods**:
   - **Neighbor**: Symbol-rate sampling (7 neighboring symbol instants)
   - **Fractional**: Inter-symbol fractional sampling
   - **Hybrid**: Combination of symbol + fractional samples
   - **Structured CNN**: 7×7 matrix structure for spatial ISI processing

3. **Neural Network-Based Equalization**:
   - Fully connected networks for Neighbor/Fractional/Hybrid
   - Convolutional neural network (CNN) for Structured approach
   - Trained to handle nonlinear distortion from PA saturation

---

## Files in This Package

| File | Purpose |
|------|---------|
| `ftn_nn_with_impairments.m` | **Main command-line simulation** - Complete BER comparison |
| `ftn_nn_gui.m` | **Interactive GUI** - Easy configuration and testing |
| `pa_models.m` | PA saturation models (Rapp, Saleh, Soft Limiter) |
| `impairments.m` | Hardware impairment functions (legacy, optional) |
| `ftn_with_pa_saturation.m` | Legacy simple detection (symbol-rate vs fractional) |
| `ftn_sim_gui.m` | Legacy GUI for simple detection |
| `HOW_TO_USE.md` | This guide |

---

## Quick Start

### Option 1: Interactive GUI (Recommended)

```matlab
% Launch the GUI
ftn_nn_gui()
```

**Steps**:
1. Configure FTN parameters (tau, beta, sps, span) in the **Configuration** tab
2. Enable/disable hardware impairments and set their parameters
3. Select detection approaches to compare (check/uncheck boxes)
4. Switch to **Training & Testing** tab
5. Click "**Start Simulation**" button
6. Monitor progress in real-time
7. View results in the **Results** tab
8. Click "**Save Results**" to export data

### Option 2: Command-Line Simulation

```matlab
% Run the main simulation script
ftn_nn_with_impairments
```

This will:
- Generate training data with PA saturation
- Train 4 neural network detectors
- Test over SNR range 0-14 dB (step 2 dB)
- Display BER curves and save results

---

## Configuration Guide

### FTN Parameters

| Parameter | Symbol | Description | Typical Values |
|-----------|--------|-------------|----------------|
| **Tau** | τ | FTN compression factor | 0.7 (30% faster than Nyquist) |
| **Beta** | β | SRRC roll-off factor | 0.3 |
| **SPS** | - | Samples per symbol | 10 |
| **Span** | - | Pulse span (symbols) | 6 |

### PA Saturation Models

#### 1. Rapp Model (SSPA)
```matlab
config.pa_model = 'rapp';
config.pa_ibo_db = 3;  % Input Back-Off (dB)
```
- **Best for**: Solid-state power amplifiers
- **IBO**: Controls saturation level (higher = less saturation)
- **Typical range**: 2-6 dB

#### 2. Saleh Model (TWTA)
```matlab
config.pa_model = 'saleh';
config.pa_ibo_db = 3;
```
- **Best for**: Traveling-wave tube amplifiers
- **Features**: Both AM/AM and AM/PM distortion
- **Typical range**: 2-5 dB

#### 3. Soft Limiter
```matlab
config.pa_model = 'soft_limiter';
config.pa_ibo_db = 3;
```
- **Best for**: Simple clipping model
- **Features**: Piecewise linear saturation
- **Typical range**: 1-4 dB

### Other Hardware Impairments

#### TX IQ Imbalance
```matlab
config.iq_tx_enabled = true;
config.iq_tx_amp = 0.1;      % Amplitude imbalance (0-0.5)
config.iq_tx_phase = 5;      % Phase imbalance (degrees, 0-30)
```

#### Phase Noise
```matlab
config.phase_noise_enabled = true;
config.pn_variance = 0.01;   % Variance (0.001-0.1)
```

#### Carrier Frequency Offset (CFO)
```matlab
config.cfo_enabled = true;
config.cfo_hz = 100;         % Offset in Hz
```

---

## Detection Approaches Explained

### 1. Neighbor (Symbol-Rate Sampling)

**Concept**: Sample matched filter output at 7 neighboring symbol instants.

**Offsets**: `[-3T, -2T, -T, 0, T, 2T, 3T]` where T = τ × sps

**Advantages**:
- Classical approach
- Baseline for comparison

**Disadvantages**:
- Loses information about nonlinear distortion between symbols
- Suboptimal for PA saturation

### 2. Fractional (Inter-Symbol Sampling)

**Concept**: Sample between symbol instants, avoiding exact symbol times.

**Offsets**: Evenly spaced within inter-symbol intervals

**Advantages**:
- Captures nonlinear distortion better
- More robust to PA saturation

**Disadvantages**:
- Loses direct symbol information

### 3. Hybrid

**Concept**: Combines symbol instants with fractional samples.

**Offsets**: `[-T, -2T/3, -T/3, 0, T/3, 2T/3, T]`

**Advantages**:
- Best of both worlds
- Balances symbol + distortion information

**Disadvantages**:
- Slightly more complex

### 4. Structured CNN

**Concept**: Extracts 7×7 matrix (7 neighbor symbols × 7 samples around each), processes with CNN.

**Matrix Structure**:
```
Row 1: [samples around symbol k-3]
Row 2: [samples around symbol k-2]
Row 3: [samples around symbol k-1]
Row 4: [samples around symbol k  ] ← Current symbol
Row 5: [samples around symbol k+1]
Row 6: [samples around symbol k+2]
Row 7: [samples around symbol k+3]
```

**Advantages**:
- Spatial processing of ISI structure
- CNN learns 2D patterns in ISI
- **Best performance** with nonlinearities

**Disadvantages**:
- Higher computational cost
- Requires more training data

---

## Usage Examples

### Example 1: Compare All Detectors with PA Saturation

```matlab
% Edit ftn_nn_with_impairments.m configuration section:
tau = 0.7;
config.pa_enabled = true;
config.pa_model = 'rapp';
config.pa_ibo_db = 3;

% Run
ftn_nn_with_impairments
```

**Expected output**: Structured CNN performs best, followed by Hybrid, then Fractional, then Neighbor.

### Example 2: Test Without Impairments (Baseline)

```matlab
% Edit configuration:
config.pa_enabled = false;
config.iq_tx_enabled = false;
config.phase_noise_enabled = false;
config.cfo_enabled = false;

% Run
ftn_nn_with_impairments
```

**Expected output**: All approaches perform similarly (linear channel).

### Example 3: Severe PA Saturation

```matlab
% Edit configuration:
config.pa_enabled = true;
config.pa_model = 'rapp';
config.pa_ibo_db = 1;  % Very low IBO = severe saturation

% Run
ftn_nn_with_impairments
```

**Expected output**: Large performance gap between Structured CNN and Neighbor.

### Example 4: Multiple Impairments

```matlab
% Edit configuration:
config.pa_enabled = true;
config.pa_ibo_db = 3;
config.iq_tx_enabled = true;
config.iq_tx_amp = 0.1;
config.iq_tx_phase = 5;
config.phase_noise_enabled = true;
config.pn_variance = 0.01;

% Run
ftn_nn_with_impairments
```

**Expected output**: Tests robustness to combined impairments.

### Example 5: Custom SNR Range

```matlab
% Edit configuration:
SNR_train = 12;           % Higher training SNR
SNR_test = 4:2:16;        % Different test range

% Run
ftn_nn_with_impairments
```

---

## Understanding the Results

### BER Curves

The simulation generates a plot showing BER vs SNR for each detection approach.

**How to interpret**:
- **Lower curve = better performance**
- **Structured CNN** should be lowest (best)
- **Neighbor** should be highest (worst) with PA saturation
- **Gap between curves** = benefit of advanced detection

### Performance Metrics

At high SNR (e.g., 14 dB), typical BER values:

| Approach | Without PA | With PA (IBO=3dB) |
|----------|-----------|-------------------|
| Neighbor | ~1e-4 | ~5e-2 |
| Fractional | ~1e-4 | ~1e-2 |
| Hybrid | ~1e-4 | ~5e-3 |
| Structured CNN | ~1e-4 | ~1e-3 |

**Key observation**: PA saturation degrades Neighbor significantly, but Structured CNN remains robust.

---

## Troubleshooting

### Problem: BER stuck at 0.5 (random guessing)

**Causes**:
1. Incorrect symbol timing
2. Mismatched training/testing conditions
3. Insufficient training data

**Solutions**:
- Check that `symbol_indices = delay + 1 + (0:N-1) * step` (note the +1)
- Ensure `conv(., ., 'full')` is used (not 'same')
- Increase N_train to 50,000+

### Problem: "Out of memory" error

**Solutions**:
- Reduce N_train (try 30,000)
- Reduce N_test (try 10,000)
- Reduce max_epochs (try 20)
- Close other MATLAB figures/variables

### Problem: Training takes too long

**Solutions**:
- Reduce max_epochs (30 → 20)
- Increase mini_batch (512 → 1024)
- Disable some detection approaches
- Use GPU (if available): `trainNetwork(... 'ExecutionEnvironment', 'gpu')`

### Problem: Poor BER performance

**Check**:
1. PA is actually enabled: `config.pa_enabled = true`
2. Training SNR is reasonable (8-12 dB recommended)
3. Neural network trained successfully (no NaN losses)
4. Sufficient training data (N_train ≥ 50,000)

### Problem: GUI doesn't start

**Possible causes**:
- MATLAB version too old (requires R2020a+)
- Missing Deep Learning Toolbox

**Solutions**:
- Update MATLAB
- Install Deep Learning Toolbox: `matlab.addons.install('Deep Learning Toolbox')`
- Use command-line version instead

---

## Advanced Customization

### Modify Neural Network Architecture

Edit the `train_nn()` function in `ftn_nn_with_impairments.m`:

```matlab
function net = train_nn(X, y, hidden_sizes, max_epochs, mini_batch)
    layers = [
        featureInputLayer(size(X,2))
        fullyConnectedLayer(64)        % Changed from 32
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.3)              % Changed from 0.2
        fullyConnectedLayer(32)        % Changed from 16
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];
    % ... rest of function
end
```

### Modify CNN Architecture

Edit the `train_cnn()` function:

```matlab
layers = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    convolution2dLayer([1 7], 64, 'Padding', 'same')   % More filters
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([7 1], 32, 'Padding', 0)        % More filters
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];
```

---

## Performance Tips

### For Faster Simulations

1. **Reduce data size**:
   ```matlab
   N_train = 30000;    % Instead of 50000
   N_test = 10000;     % Instead of 20000
   ```

2. **Fewer SNR points**:
   ```matlab
   SNR_test = 0:4:16;  % Step 4 instead of 2
   ```

3. **Fewer epochs**:
   ```matlab
   max_epochs = 20;    % Instead of 30
   ```

4. **Disable approaches**:
   - Test only Neighbor + Structured CNN
   - Skip Fractional and Hybrid

### For Better Accuracy

1. **More training data**:
   ```matlab
   N_train = 100000;   % More data
   ```

2. **More epochs**:
   ```matlab
   max_epochs = 50;
   ```

3. **Better training SNR**:
   ```matlab
   SNR_train = 12;     % Higher SNR for cleaner training
   ```

---

## Citation and References

If you use this code in your research, please cite:

**Professor's Research Context**:
> "Symbol-rate sampling provides sufficient statistics for linear channels only. With nonlinearities (e.g., PA saturation), fractional sampling and window-based detection capture distortion better than symbol-rate sampling."
>
> — Prof. Dr. Enver Çavuş

**Key Concept**: This implementation validates that neural network-based window detectors (especially Structured CNN) outperform classical symbol-rate sampling when PA saturation is present.

---

## Quick Reference: Parameter Ranges

| Parameter | Min | Typical | Max | Notes |
|-----------|-----|---------|-----|-------|
| **tau** | 0.5 | 0.7 | 0.99 | Lower = more ISI |
| **beta** | 0.1 | 0.3 | 0.9 | Roll-off factor |
| **sps** | 4 | 10 | 20 | Higher = more accuracy |
| **PA IBO** | 0.5 | 3 | 10 | Lower = more saturation |
| **N_train** | 10k | 50k | 200k | More = better learning |
| **max_epochs** | 10 | 30 | 100 | More = better training |

---

**Author**: Emre Cerci
**Supervisor**: Prof. Dr. Enver Çavuş
**Institution**: Atilim University
**Date**: January 2026

---

**END OF GUIDE**
