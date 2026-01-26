## 🚀 HOW TO USE - Python FTN Simulation

### Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run simulation (no impairments - baseline)
python ftn_sim_configurable.py

# 3. Done! Check ber_results.png
```

---

## 📦 Installation

### Option 1: Using pip (Recommended)
```bash
cd python/pa_saturation
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda create -n ftn python=3.9
conda activate ftn
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import numpy, torch, matplotlib; print('✓ All dependencies OK')"
```

---

## 🎮 Usage Examples

### 1. **Baseline (No Impairments)**
```bash
python ftn_sim_configurable.py
```
**What it does:** Runs FTN simulation without any hardware impairments. Good for reference performance.

### 2. **PA Saturation Only**
```bash
python ftn_sim_configurable.py --pa
```
**What it does:** Enables PA saturation with default settings (Rapp model, IBO=3dB)

### 3. **PA Saturation with Custom Settings**
```bash
python ftn_sim_configurable.py --pa --pa-model rapp --pa-ibo 5
```
**Options:**
- `--pa-model`: `rapp`, `saleh`, or `soft_limiter`
- `--pa-ibo`: Input Back-Off in dB (1-10, lower = more saturation)
- `--pa-memory`: Add memory effects to PA

### 4. **PA + IQ Imbalance**
```bash
python ftn_sim_configurable.py --pa --iq-tx
```
**What it does:** Adds transmitter IQ imbalance on top of PA saturation

**Customize IQ imbalance:**
```bash
python ftn_sim_configurable.py --pa --iq-tx --iq-tx-amp 1.0 --iq-tx-phase 10
```
- `--iq-tx-amp`: Amplitude imbalance in dB (typical: 0.1-2)
- `--iq-tx-phase`: Phase imbalance in degrees (typical: 1-15)

### 5. **PA + Phase Noise**
```bash
python ftn_sim_configurable.py --pa --pn
```
**Customize:**
```bash
python ftn_sim_configurable.py --pa --pn --pn-psd -70
```
- `--pn-psd`: Phase noise PSD in dBc/Hz (typical: -60 to -100)

### 6. **PA + Quantization**
```bash
python ftn_sim_configurable.py --pa --dac --adc
```
**Customize:**
```bash
python ftn_sim_configurable.py --pa --dac --dac-bits 6 --adc --adc-bits 6
```
- `--dac-bits`: DAC resolution (typical: 4-12 bits)
- `--adc-bits`: ADC resolution (typical: 4-12 bits)

### 7. **PA + CFO**
```bash
python ftn_sim_configurable.py --pa --cfo
```
**Customize:**
```bash
python ftn_sim_configurable.py --pa --cfo --cfo-hz 200
```
- `--cfo-hz`: Carrier frequency offset in Hz

### 8. **ALL Impairments**
```bash
python ftn_sim_configurable.py --all
```
**What it does:** Enables all impairments with moderate default values:
- PA saturation (Rapp, IBO=3dB)
- TX IQ imbalance (0.5dB, 5°)
- Phase noise (-80 dBc/Hz)
- DAC/ADC quantization (8-bit)
- CFO (50 Hz)
- RX IQ imbalance (0.3dB, 3°)

### 9. **Custom FTN Parameters**
```bash
python ftn_sim_configurable.py --pa --tau 0.8 --l-frac 4
```
**Options:**
- `--tau`: FTN compression (0.5-0.9, lower = more ISI)
- `--l-frac`: Fractional sampling factor (2, 4, 8 for T/2, T/4, T/8)

### 10. **Save/Load Configuration**

**Save configuration:**
```bash
python ftn_sim_configurable.py --pa --iq-tx --pn --save-config my_setup.json
```

**Load and run:**
```bash
python ftn_sim_configurable.py --load my_setup.json
```

---

## 🎛️ All Command-Line Options

### Impairment Toggles

| Option | Description | Default |
|--------|-------------|---------|
| `--all` | Enable all impairments | OFF |
| `--pa` | Enable PA saturation | OFF |
| `--iq-tx` | Enable TX IQ imbalance | OFF |
| `--iq-rx` | Enable RX IQ imbalance | OFF |
| `--pn` | Enable phase noise | OFF |
| `--dac` | Enable DAC quantization | OFF |
| `--adc` | Enable ADC quantization | OFF |
| `--cfo` | Enable CFO | OFF |

### PA Saturation Parameters

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--pa-model` | PA model type | `rapp` | `rapp`, `saleh`, `soft_limiter` |
| `--pa-ibo` | Input Back-Off (dB) | 3 | 1-10 |
| `--pa-memory` | Enable memory effects | OFF | flag |

### IQ Imbalance Parameters

| Option | Description | Default | Typical Range |
|--------|-------------|---------|---------------|
| `--iq-tx-amp` | TX amplitude imbalance (dB) | 0.5 | 0.1-2.0 |
| `--iq-tx-phase` | TX phase imbalance (deg) | 5 | 1-15 |
| `--iq-rx-amp` | RX amplitude imbalance (dB) | 0.3 | 0.1-1.5 |
| `--iq-rx-phase` | RX phase imbalance (deg) | 3 | 1-10 |

### Other Impairment Parameters

| Option | Description | Default | Typical Range |
|--------|-------------|---------|---------------|
| `--pn-psd` | Phase noise PSD (dBc/Hz) | -80 | -60 to -100 |
| `--dac-bits` | DAC resolution (bits) | 8 | 4-12 |
| `--adc-bits` | ADC resolution (bits) | 8 | 4-12 |
| `--cfo-hz` | CFO in Hz | 50 | 10-1000 |

### FTN Parameters

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--tau` | FTN compression factor | 0.7 | 0.5-0.9 |
| `--l-frac` | Fractional sampling factor | 2 | 2, 4, 8 |

### Simulation Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--n-train` | Training symbols | 50000 |
| `--n-test` | Test symbols per SNR | 20000 |
| `--epochs` | NN training epochs | 15 |

### Configuration Management

| Option | Description |
|--------|-------------|
| `--save-config FILE` | Save configuration to JSON file |
| `--load FILE` | Load configuration from JSON file |

---

## 📊 Output Files

After running, you'll get:

1. **`ber_results.png`** - BER vs SNR plot
2. **`simulation_results.json`** - Complete results in JSON format
3. **Console output** - Real-time progress and BER table

---

## 🧪 Test Individual Impairments

### Test PA Models
```bash
python pa_models.py
```
**Output:** `pa_models_comparison.png` showing AM/AM characteristics

### Test All Impairments Visually
```bash
python impairments.py
```
**Output:** `impairments_test.png` showing constellation diagrams

---

## 💡 Practical Scenarios

### Scenario 1: Study PA Saturation Impact
```bash
# Baseline
python ftn_sim_configurable.py --save-config baseline.json

# Light saturation
python ftn_sim_configurable.py --pa --pa-ibo 6 --save-config pa_light.json

# Heavy saturation
python ftn_sim_configurable.py --pa --pa-ibo 2 --save-config pa_heavy.json

# Compare results!
```

### Scenario 2: Compare Fractional Sampling Benefits
```bash
# T-spaced (symbol rate) - add this manually by setting l_frac=1
python ftn_sim_configurable.py --pa --l-frac 1

# T/2-spaced (fractional)
python ftn_sim_configurable.py --pa --l-frac 2

# T/4-spaced (more fractional)
python ftn_sim_configurable.py --pa --l-frac 4
```

### Scenario 3: Realistic Channel
```bash
# Real-world hardware: PA + IQ + phase noise + quantization
python ftn_sim_configurable.py --pa --iq-tx --iq-rx --pn --dac --adc
```

### Scenario 4: Worst Case
```bash
# All impairments + aggressive FTN
python ftn_sim_configurable.py --all --tau 0.6 --pa-ibo 2
```

---

## 🐛 Troubleshooting

### Problem: "No module named 'torch'"
**Solution:**
```bash
pip install torch
# Or for CPU-only (smaller):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: "CUDA out of memory"
**Solution:** Using CPU is fine for this simulation
```bash
# PyTorch automatically falls back to CPU if CUDA unavailable
python ftn_sim_configurable.py  # Will use CPU if no GPU
```

### Problem: Simulation too slow
**Solutions:**
```bash
# 1. Reduce training/test samples
python ftn_sim_configurable.py --pa --n-train 20000 --n-test 10000

# 2. Reduce epochs
python ftn_sim_configurable.py --pa --epochs 10

# 3. Use GPU if available (automatic)
```

### Problem: Results not reproducible
**Solution:** Add random seed (modify code):
```python
# Add to beginning of ftn_sim_configurable.py
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
```

---

## 📚 Understanding the Output

### Console Output Example
```
======================================================================
 FTN SIMULATION WITH CONFIGURABLE IMPAIRMENTS
======================================================================

FTN Parameters: tau=0.7, beta=0.35, sps=10
Fractional sampling: L=2 (T/2 spacing)

============================================================
Impairment Configuration
============================================================

[Transmitter Impairments]
  ✓ PA Saturation: rapp, IBO=3dB
  ✗ TX IQ Imbalance: OFF
  ✗ DAC Quantization: OFF

[Receiver Impairments]
  ✗ CFO: OFF
  ✗ Phase Noise: OFF
  ✗ RX IQ Imbalance: OFF
  ✗ ADC Quantization: OFF
============================================================

Using device: cpu

[1/4] Generating training data...
[2/4] Extracting features...
Training samples: 49850, Feature dimension: 34
[3/4] Training neural network equalizer...
  Epoch 5/15, Loss: 0.182453
  Epoch 10/15, Loss: 0.091247
  Epoch 15/15, Loss: 0.067832

[4/4] Testing BER performance...
======================================================================
SNR (dB)   | BER
----------------------------------------------------------------------
0          | 2.45e-02
2          | 1.12e-02
4          | 3.87e-03
6          | 9.23e-04
8          | 1.45e-04
10         | 1.82e-05
12         | 2.10e-06
14         | 1.50e-07
======================================================================

Results saved to: simulation_results.json
BER plot saved: ber_results.png
```

### Understanding BER Values
- **> 1e-2 (0.01)**: Poor performance, lots of errors
- **1e-3 to 1e-2**: Moderate, acceptable for some applications
- **1e-4 to 1e-3**: Good performance
- **< 1e-5**: Excellent performance

---

## 🎓 Tips for Research

### 1. **Systematic Study**
```bash
# Create a batch script to run multiple configurations
for ibo in 2 3 4 5; do
    python ftn_sim_configurable.py --pa --pa-ibo $ibo --save-config pa_ibo_${ibo}.json
done
```

### 2. **Compare Impairment Effects**
Run simulations with one impairment at a time to isolate effects:
```bash
python ftn_sim_configurable.py                    # Baseline
python ftn_sim_configurable.py --pa              # + PA only
python ftn_sim_configurable.py --pa --iq-tx      # + PA + IQ
python ftn_sim_configurable.py --pa --iq-tx --pn # + PA + IQ + Phase noise
```

### 3. **Document Everything**
Save configurations for reproducibility:
```bash
python ftn_sim_configurable.py --pa --iq-tx --save-config experiment_001.json
```

### 4. **Visualize Results**
All plots are saved automatically. You can also modify the code to save more detailed plots.

---

## 📖 Related Files

- **`pa_models.py`** - PA saturation models (Rapp, Saleh, Soft Limiter)
- **`impairments.py`** - All hardware impairment implementations
- **`ftn_sim_configurable.py`** - Main simulation (this file)
- **`ftn_with_pa_saturation.py`** - Original simpler version
- **`README.md`** - Project overview and theory

---

## 🤝 Getting Help

1. **Check error messages** - Usually self-explanatory
2. **Read README.md** - Detailed theory and background
3. **Try simpler configurations** - Start with `--pa` only
4. **Test individual components** - Run `python impairments.py` to test

---

## ⚡ Quick Reference Card

```bash
# Most Common Commands

# 1. Baseline
python ftn_sim_configurable.py

# 2. PA only
python ftn_sim_configurable.py --pa

# 3. PA + IQ
python ftn_sim_configurable.py --pa --iq-tx

# 4. Everything
python ftn_sim_configurable.py --all

# 5. Save config
python ftn_sim_configurable.py --pa --save-config my.json

# 6. Load config
python ftn_sim_configurable.py --load my.json
```

---

**Happy Simulating! 🚀**

For questions, check README.md or examine the code comments.
