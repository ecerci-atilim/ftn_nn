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
Power Amplifier nonlinearity models:

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
- Symbol-rate sampling detector
- Fractional sampling detector (T/L spacing)
- BER comparison
- Performance visualization

## Usage

```matlab
% Run main simulation
cd matlab/pa_saturation
ftn_with_pa_saturation
```

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

- **Symbol-rate sampling**: Higher BER due to insufficient statistics
- **Fractional sampling**: Lower BER (2-4 dB gain typical)
- **Gain increases** with:
  - Stronger PA saturation (lower IBO)
  - Higher compression (lower τ)
  - More fractional samples (higher L)

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
