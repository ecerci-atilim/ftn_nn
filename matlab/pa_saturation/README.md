# FTN with PA Saturation - MATLAB Implementation

## Overview

This MATLAB implementation demonstrates the advantage of **Fractionally Spaced Equalization (FSE)** over symbol-rate sampling in Faster-than-Nyquist (FTN) signaling systems with **Power Amplifier (PA) saturation nonlinearity**.

## Key Concept

### Why Fractional Sampling Helps with PA Nonlinearity

1. **Symbol-rate sampling sufficiency theorem** assumes a **linear** band-limited channel
2. When PA saturation introduces **nonlinearity**, symbol-rate samples are **no longer sufficient statistics**
3. **Fractional sampling** (e.g., T/2 spacing) captures:
   - Nonlinear ISI patterns
   - Spectral regrowth from PA
   - Amplitude-dependent distortion
4. More samples → Better nonlinearity mitigation

## System Model

```
Bits → BPSK → FTN (SRRC, τ<1) → PA Saturation → AWGN Channel (Linear)
                                       ↓
                                  NONLINEARITY
                                  (Key Point!)
                                       ↓
     ← Detection ← Sampling ← Matched Filter (SRRC)
                      ↓
            Symbol-Rate vs Fractional
```

## Files

### `pa_models.m`
Power Amplifier nonlinearity models with numerical safeguards:

1. **Rapp Model** (Solid State PA)
   - `y(r) = G·r / [1 + (r/Asat)^(2p)]^(1/2p)`
   - Smooth saturation, no phase distortion
   - Typical for SSPAs

2. **Saleh Model** (Traveling Wave Tube Amplifier)
   - AM/AM: `A(r) = αa·r / (1 + βa·r²)`
   - AM/PM: `Φ(r) = αp·r² / (1 + βp·r²)`
   - Models both amplitude and phase distortion
   - Typical for TWTAs

3. **Soft Limiter**
   - Linear → Compression → Saturation
   - Simple piecewise model

### `ftn_with_pa_saturation.m`
Main simulation script:

- FTN signal generation with PA saturation
- Correct SNR calculation based on actual signal power
- Multiple detection approaches
- BER comparison and visualization

### `train_nn_models.m`
Train multiple NN architectures for comparison:
- FC variants (Shallow, Standard, Deep, Wide, Bottleneck)
- 1D-CNN variants
- 2D-CNN structured variants (7x7 matrix)
- LSTM/BiLSTM/GRU recurrent networks
- Different optimizers (Adam, SGDM, RMSProp)

### `train_nn_models_optimized.m` ⭐ NEW
**Optimized training** with all performance improvements:
- **Decision Feedback** (D=4) for significant BER improvement
- **Hybrid Sampling** capturing both symbol and ISI information
- **Multi-SNR Training** for better generalization
- **Correct SNR/Noise Model** based on actual signal power
- Training at SNR=8dB for improved decision boundaries
- Consistent normalization across training and testing

### `test_nn_models.m`
Test trained models with file picker UI:
- Multi-file selection
- Supports all model types (FC, CNN1D, CNN2D, LSTM, GRU)
- Supports Decision Feedback models (fc_df, lstm_df)
- BER comparison plots and bar charts

### `test_nn_models_optimized.m` ⭐ NEW
**Optimized testing** with adaptive block testing:
- Minimum errors threshold for statistically reliable BER
- Maximum symbols cap for efficiency
- Sequential Decision Feedback detection
- Uses training normalization parameters

## Usage

```matlab
% Run main simulation
cd matlab/pa_saturation
ftn_with_pa_saturation

% Train optimized models (RECOMMENDED)
train_nn_models_optimized

% Test models with file picker
test_nn_models_optimized
```

## Optimizations Implemented

### 1. Signal Processing Fixes
- **Correct SNR calculation**: Uses actual signal power instead of assuming unit energy
- **Real signal path**: Forces real output for BPSK after PA (PA may output complex)
- **Optimized memory**: Exact size allocation for upsampled signal

### 2. Decision Feedback (D=4)
The most significant improvement. During training, uses true bits (teacher forcing).
During testing, uses previous decisions sequentially.
- Input: 7 samples + 4 feedback bits = 11 features
- Expected improvement: ~5-10x better BER

### 3. Hybrid Sampling
Captures both symbol instants and inter-symbol information:
```matlab
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets_hybrid = [-step, -t2, -t1, 0, t1, t2, step];
```

### 4. Multi-SNR Training
Trains with data from multiple SNR values for better generalization:
```matlab
SNR_train_range = [6, 8, 10];  % dB
```

### 5. Training at Lower SNR
Uses SNR=8dB instead of 10dB, forcing the network to learn better decision boundaries.

## Configuration Parameters

```matlab
% FTN Parameters
tau = 0.7;              % Compression factor (0.7 = 30% faster)
beta = 0.35;            % SRRC roll-off
sps = 10;               % Samples per symbol

% PA Saturation
PA_MODEL = 'rapp';      % 'rapp', 'saleh', 'soft_limiter'
IBO_dB = 3;             % Input Back-Off (controls saturation)

% Fractional Sampling
L_frac = 2;             % T/2 spacing (can be 2, 4, 8...)
```

## Expected Results

### Target Performance (τ=0.7, 10dB SNR)
- **Threshold detector**: ~5-7×10⁻² (baseline)
- **Neighbor NN**: ~4×10⁻³
- **Hybrid + DF**: ~1×10⁻⁵ (TARGET)

### Performance Improvements
| Optimization | Expected BER Improvement |
|--------------|-------------------------|
| Fix noise model | ~1.5x |
| Decision Feedback (D=4) | ~5-10x |
| Hybrid sampling | ~1.5-2x |
| Training SNR=8dB | ~1.2x |
| Combined | ~10-20x |

### General Guidelines
- **Symbol-rate sampling**: Higher BER due to insufficient statistics
- **Fractional sampling**: Lower BER (2-4 dB gain typical)
- **Gain increases** with:
  - Stronger PA saturation (lower IBO)
  - Higher compression (lower τ)
  - Decision Feedback depth

## Theoretical Background

### Why Symbol-Rate Sampling Fails

From sampling theorem:
- **Linear channel**: Symbol-rate samples contain all information
- **Nonlinear PA**: Creates out-of-band distortion and nonlinear ISI
- **Symbol-rate samples miss** this information → suboptimal detection

### Fractional Sampling Advantage

- Captures **spectral regrowth** from PA nonlinearity
- Observes **amplitude-dependent ISI** patterns
- Provides **more degrees of freedom** for equalization
- Approaches **sufficient statistics** as L → ∞

## References

1. Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal"
2. Saleh, A. A. (1981). "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers", IEEE Trans. Commun.
3. Tokluoğlu, R., et al. (2025). "CNN-FK3: Structured Fixed Kernel CNN for FTN Detection", IEEE Trans. Commun.

## Author

Emre Cerci
January 2026

## License

Academic/Research Use
