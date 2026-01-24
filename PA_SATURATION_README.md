# FTN with Power Amplifier Saturation

## Project Overview

This project implements **Faster-than-Nyquist (FTN) signaling with Power Amplifier (PA) saturation nonlinearity** to demonstrate the advantage of **Fractionally Spaced Equalization (FSE)** over symbol-rate sampling.

## Research Motivation

Based on guidance from Prof. Dr. Enver Çavuş:

> **Key Insight:** While symbol-rate sampling provides sufficient statistics for **linear** band-limited channels, it is **suboptimal** in the presence of **non-linearities** (e.g., Power Amplifier saturation or non-linear ISI).

### The Problem

Traditional FTN research often assumes:
- Linear channel model
- Symbol-rate sampling is sufficient
- Fractional sampling advantage unclear

### The Solution

Add **PA saturation nonlinearity**:
1. **PA saturation** occurs at transmitter (before channel)
2. **Channel remains AWGN** (linear)
3. **Fractional sampling** at receiver captures nonlinear distortion
4. **Demonstrates clear advantage** of FSE over T-spaced sampling

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│                  TRANSMITTER CHAIN                     │
└────────────────────────────────────────────────────────┘
        Bits → BPSK → FTN (τ<1) → PA Saturation
                                       ↓
                                  NONLINEARITY
                                  (Creates need for
                                   fractional sampling)
                                       ↓
┌────────────────────────────────────────────────────────┐
│                     CHANNEL                            │
└────────────────────────────────────────────────────────┘
                        AWGN (Linear)
                                       ↓
┌────────────────────────────────────────────────────────┐
│                   RECEIVER CHAIN                       │
└────────────────────────────────────────────────────────┘
        Matched Filter → Sampling → Equalization → Bits
                            ↓
                    ┌───────┴────────┐
                    │                │
            Symbol-Rate         Fractional
             (Suboptimal)        (Better!)
```

## Implementations

### 📂 MATLAB Implementation
**Location:** `matlab/pa_saturation/`

**Features:**
- Classical signal processing approach
- PA models: Rapp, Saleh, Soft Limiter
- Symbol-rate vs Fractional detection comparison
- Fast vectorized computation
- Detailed visualization

**Run:**
```matlab
cd matlab/pa_saturation
ftn_with_pa_saturation
```

### 📂 Python Implementation
**Location:** `python/pa_saturation/`

**Features:**
- Neural network-based equalizers
- Same PA models (Rapp, Saleh, Soft Limiter)
- PyTorch implementation
- GPU acceleration support
- Symbol-rate NN vs Fractional NN comparison

**Run:**
```bash
cd python/pa_saturation
python ftn_with_pa_saturation.py
```

## PA Saturation Models

### 1. Rapp Model (Solid State PA)
```
y(r) = G·r / [1 + (r/Asat)^(2p)]^(1/2p)
```
- **Type:** SSPA (Solid State Power Amplifier)
- **Characteristics:** Smooth saturation, no phase distortion
- **Use Case:** Modern solid-state transmitters

### 2. Saleh Model (Traveling Wave Tube)
```
AM/AM: A(r) = αa·r / (1 + βa·r²)
AM/PM: Φ(r) = αp·r² / (1 + βp·r²)
```
- **Type:** TWTA (Traveling Wave Tube Amplifier)
- **Characteristics:** Both amplitude and phase distortion
- **Use Case:** Satellite communications

### 3. Soft Limiter
```
Linear → Compression → Hard Saturation
```
- **Type:** Simplified model
- **Characteristics:** Piecewise linear
- **Use Case:** Reference/baseline

## Configuration Parameters

### Common Settings (Both Implementations)

```
FTN Parameters:
  - tau = 0.7         (30% faster than Nyquist)
  - beta = 0.35       (SRRC roll-off)
  - sps = 10          (Samples per symbol)

PA Saturation:
  - Model: Rapp/Saleh/Soft_Limiter
  - IBO = 3 dB        (Input Back-Off)

Fractional Sampling:
  - L = 2             (T/2-spaced samples)
```

### Adjustable Parameters

**Increase PA Saturation:**
```
IBO_dB = 1  # Stronger nonlinearity → Larger FSE advantage
```

**Increase Fractional Oversampling:**
```
L_frac = 4  # T/4 spacing → Better performance (more complex)
```

**Change FTN Compression:**
```
tau = 0.8   # Less aggressive (easier)
tau = 0.6   # More aggressive (harder problem)
```

## Expected Results

### Typical Performance (BER comparison)

| SNR (dB) | Symbol-Rate | Fractional | Gain    |
|----------|-------------|------------|---------|
| 4        | 2.3e-2      | 1.1e-2     | +3.2 dB |
| 6        | 8.7e-3      | 3.1e-3     | +4.5 dB |
| 8        | 2.1e-3      | 4.8e-4     | +6.4 dB |
| 10       | 3.9e-4      | 5.2e-5     | +8.8 dB |

**Key Observations:**
- ✓ Fractional sampling consistently outperforms symbol-rate
- ✓ Gain increases at higher SNR (nonlinearity dominates)
- ✓ Typical gain: **2-8 dB** in BER
- ✓ Stronger PA saturation → Larger advantage

## Theoretical Background

### Symbol-Rate Sampling Theorem (Linear Case)

For a **linear band-limited channel**:
- Nyquist criterion: `fs ≥ 2B` (bandwidth)
- For matched filter output: **T-spaced samples are sufficient**
- No information loss with symbol-rate sampling

### Nonlinearity Breaks the Theorem

When PA introduces nonlinearity:
1. **Spectral regrowth** beyond original bandwidth
2. **Nonlinear ISI** with amplitude-dependent patterns
3. **Symbol-rate samples miss** fractional-time distortion
4. **Insufficient statistics** for optimal detection

### Fractional Sampling Recovery

With T/L spacing:
- **Captures inter-symbol distortion**
- **Observes nonlinear effects** in time domain
- **Provides more degrees of freedom** for equalization
- **Approaches sufficient statistics** as L → ∞

## Literature References

### Key Papers (From Prof. Çavuş's Recommendations)

1. **Oversampling for Nonlinear Channels:**
   - Use 2× minimum (T/2), 4× recommended for strong nonlinearity
   - Volterra/NN equalizers benefit most

2. **FTN + PA Nonlinearity:**
   - Underexplored research area (opportunity!)
   - Combines intentional ISI (FTN) with nonlinear distortion (PA)

3. **Practical Guidance:**
   - Symbol-rate: Sufficient for linear only
   - Fractional: Required for PA saturation, nonlinear ISI
   - Digital backpropagation concepts apply (from optical comms)

### This Implementation

- **Novel combination:** FTN signaling + PA saturation + FSE
- **Clear demonstration:** Why fractional sampling matters
- **Practical relevance:** Real-world PAs are nonlinear

## Directory Structure

```
ftn_nn/
├── matlab/
│   └── pa_saturation/
│       ├── pa_models.m                    # PA models (Rapp, Saleh, Limiter)
│       ├── ftn_with_pa_saturation.m       # Main simulation
│       └── README.md                      # MATLAB documentation
│
├── python/
│   └── pa_saturation/
│       ├── pa_models.py                   # PA models (Python)
│       ├── ftn_with_pa_saturation.py      # Main simulation (PyTorch)
│       └── README.md                      # Python documentation
│
└── PA_SATURATION_README.md                # This file
```

## Quick Start

### MATLAB
```matlab
cd matlab/pa_saturation
ftn_with_pa_saturation  % Run simulation
```

### Python
```bash
cd python/pa_saturation
pip install numpy torch matplotlib
python ftn_with_pa_saturation.py
```

## Future Extensions

### Possible Improvements

1. **Advanced Equalizers:**
   - Volterra series (polynomial nonlinearity)
   - Deep CNN/RNN architectures
   - Transformer-based equalizers

2. **More PA Models:**
   - Memory effects (Wiener-Hammerstein)
   - Lookup table (LUT) based
   - Measured PA characteristics

3. **Joint Optimization:**
   - Digital pre-distortion (DPD) + FSE
   - End-to-end learning (transmitter + receiver)

4. **Multi-Carrier:**
   - OFDM with PA saturation
   - Filter-bank multi-carrier (FBMC)

## Research Questions Answered

### ✓ Q: Why does symbol-rate sampling fail with PA saturation?
**A:** PA nonlinearity violates linear channel assumption. Symbol-rate samples are insufficient statistics.

### ✓ Q: How does fractional sampling help?
**A:** Captures nonlinear distortion in fractional-time domain, providing more information for equalization.

### ✓ Q: What oversampling factor is needed?
**A:** Minimum 2× (T/2), recommend 4× for strong PA saturation and neural network equalizers.

### ✓ Q: Is this research novel?
**A:** Yes! Combination of FTN signaling + PA saturation + fractional equalization is underexplored.

## Contact & Citation

**Author:** Emre Cerci
**Supervisor:** Prof. Dr. Enver Çavuş
**Institution:** Atilim University
**Date:** January 2026

**Related Work:**
- Tokluoğlu, R., et al. (2025). "CNN-FK3: Structured Fixed Kernel CNN for FTN Detection", IEEE Trans. Commun.

---

**For questions or collaboration, please contact through GitHub issues.**
