# 🚀 HOW TO USE - MATLAB FTN Simulation

## Quick Start (2 Steps)

```matlab
% 1. Open MATLAB and navigate to folder
cd matlab/pa_saturation

% 2. Run GUI
ftn_sim_gui
```

Or run directly from command line:
```matlab
% Run simulation with PA saturation only
ftn_with_pa_saturation
```

---

## 📦 Installation

### Requirements
- **MATLAB R2018b or later** (recommended)
- **No additional toolboxes required** (uses base MATLAB only)

### Verify Installation
```matlab
% Check MATLAB version
version

% Test that files are accessible
which pa_models
which impairments
which ftn_sim_gui
```

---

## 🎮 Usage Methods

### Method 1: Interactive GUI ⭐ (Recommended for Beginners)

```matlab
ftn_sim_gui
```

**Features:**
- Point-and-click interface
- Toggle impairments on/off with checkboxes
- Adjust parameters with text boxes
- Quick presets (All OFF, PA Only, All ON)
- Real-time status updates

**How to Use:**
1. Launch GUI: `ftn_sim_gui`
2. Set FTN parameters (tau, L_frac, N_train)
3. Check/uncheck impairments you want
4. Click **RUN SIMULATION**
5. View results in new figure window

---

### Method 2: Command-Line Scripts

#### Example 1: Baseline (No Impairments)
```matlab
% Run original simulation
ftn_with_pa_saturation
```

#### Example 2: Test Individual Impairments

**PA Saturation Only:**
```matlab
% Create configuration
config = struct();
config.pa_saturation.enabled = true;
config.pa_saturation.model = 'rapp';
config.pa_saturation.IBO_dB = 3;

% Other impairments OFF
config.iq_imbalance_tx.enabled = false;
config.dac_quantization.enabled = false;
% ... (set others to false)

% Run simulation
run_ftn_with_config(config);
```

**PA + IQ Imbalance:**
```matlab
config = struct();

% Enable PA
config.pa_saturation.enabled = true;
config.pa_saturation.model = 'rapp';
config.pa_saturation.IBO_dB = 3;
config.pa_saturation.G = 1;
config.pa_saturation.Asat = sqrt(10^(3/10));
config.pa_saturation.p = 2;

% Enable TX IQ imbalance
config.iq_imbalance_tx.enabled = true;
config.iq_imbalance_tx.amp_dB = 0.5;
config.iq_imbalance_tx.phase_deg = 5;

% Disable others
config.dac_quantization.enabled = false;
config.cfo.enabled = false;
config.phase_noise.enabled = false;
config.iq_imbalance_rx.enabled = false;
config.adc_quantization.enabled = false;

% Test signal
signal = randn(1000,1) + 1j*randn(1000,1);
tx_impaired = impairments(signal, config, 'tx');

% Plot
figure;
plot(real(signal), imag(signal), 'b.', 'MarkerSize', 4);
hold on;
plot(real(tx_impaired), imag(tx_impaired), 'r.', 'MarkerSize', 4);
legend('Original', 'With Impairments');
grid on;
axis equal;
title('Signal Constellation');
```

---

### Method 3: Using Impairments Function Directly

```matlab
% Create test signal
N = 1000;
signal = exp(1j*2*pi*0.1*(1:N)');

% Configure impairments
config.pa_saturation.enabled = true;
config.pa_saturation.model = 'rapp';
config.pa_saturation.IBO_dB = 3;
config.pa_saturation.G = 1;
config.pa_saturation.Asat = sqrt(10^(3/10));
config.pa_saturation.p = 2;
config.pa_saturation.memory_effects = false;

config.iq_imbalance_tx.enabled = true;
config.iq_imbalance_tx.amp_dB = 0.5;
config.iq_imbalance_tx.phase_deg = 5;

config.dac_quantization.enabled = true;
config.dac_quantization.n_bits = 6;
config.dac_quantization.full_scale = 1.0;

% Apply TX impairments
tx_signal = impairments(signal, config, 'tx');

% Configure RX impairments
config.cfo.enabled = true;
config.cfo.cfo_hz = 100;
config.cfo.fs = 1e6;

config.phase_noise.enabled = true;
config.phase_noise.psd_dBc_Hz = -80;
config.phase_noise.fs = 1e6;

config.iq_imbalance_rx.enabled = true;
config.iq_imbalance_rx.amp_dB = 0.3;
config.iq_imbalance_rx.phase_deg = 3;

config.adc_quantization.enabled = true;
config.adc_quantization.n_bits = 6;
config.adc_quantization.full_scale = 1.0;

% Apply RX impairments
rx_signal = impairments(tx_signal, config, 'rx');

% Plot comparison
figure;
subplot(1,3,1);
plot(real(signal), imag(signal), 'b.');
title('Original'); grid on; axis equal;

subplot(1,3,2);
plot(real(tx_signal), imag(tx_signal), 'r.');
title('After TX Impairments'); grid on; axis equal;

subplot(1,3,3);
plot(real(rx_signal), imag(rx_signal), 'g.');
title('After RX Impairments'); grid on; axis equal;
```

---

## 🎛️ Configuration Parameters

### PA Saturation

```matlab
config.pa_saturation.enabled = true;          % ON/OFF
config.pa_saturation.model = 'rapp';          % 'rapp', 'saleh', 'soft_limiter'
config.pa_saturation.IBO_dB = 3;              % Input Back-Off (dB)
config.pa_saturation.memory_effects = false;  % Enable memory effects
config.pa_saturation.memory_depth = 3;        % Memory depth (if enabled)

% Model-specific parameters (auto-calculated from IBO)
IBO_lin = 10^(IBO_dB/10);
config.pa_saturation.G = 1;                   % Small signal gain
config.pa_saturation.Asat = sqrt(IBO_lin);    % Saturation amplitude
config.pa_saturation.p = 2;                   % Smoothness factor (Rapp)
```

**Typical Values:**
- `IBO_dB = 6`: Light saturation (mild nonlinearity)
- `IBO_dB = 3`: Moderate saturation (typical)
- `IBO_dB = 1`: Heavy saturation (strong nonlinearity)

### IQ Imbalance (TX/RX)

```matlab
config.iq_imbalance_tx.enabled = true;
config.iq_imbalance_tx.amp_dB = 0.5;         % Amplitude imbalance (dB)
config.iq_imbalance_tx.phase_deg = 5;        % Phase imbalance (degrees)

config.iq_imbalance_rx.enabled = true;
config.iq_imbalance_rx.amp_dB = 0.3;
config.iq_imbalance_rx.phase_deg = 3;
```

**Typical Values:**
- Good hardware: amp = 0.1-0.3 dB, phase = 1-3°
- Moderate hardware: amp = 0.3-1.0 dB, phase = 3-10°
- Poor hardware: amp = 1.0-2.0 dB, phase = 10-15°

### Quantization (DAC/ADC)

```matlab
config.dac_quantization.enabled = true;
config.dac_quantization.n_bits = 8;          % Resolution (bits)
config.dac_quantization.full_scale = 1.0;    % Full-scale value

config.adc_quantization.enabled = true;
config.adc_quantization.n_bits = 8;
config.adc_quantization.full_scale = 1.0;
```

**Typical Values:**
- High quality: 10-12 bits
- Medium quality: 8-10 bits
- Low quality: 4-8 bits

### Phase Noise

```matlab
config.phase_noise.enabled = true;
config.phase_noise.psd_dBc_Hz = -80;         % PSD in dBc/Hz
config.phase_noise.fs = 1e6;                 % Sampling frequency
```

**Typical Values:**
- Excellent oscillator: -100 to -90 dBc/Hz
- Good oscillator: -90 to -80 dBc/Hz
- Moderate oscillator: -80 to -70 dBc/Hz
- Poor oscillator: -70 to -60 dBc/Hz

### Carrier Frequency Offset

```matlab
config.cfo.enabled = true;
config.cfo.cfo_hz = 100;                     % CFO in Hz
config.cfo.fs = 1e6;                         % Sampling frequency
```

**Typical Values:**
- Depends on carrier frequency and oscillator accuracy
- For fc = 1 GHz, 10 ppm accuracy → CFO = 10 kHz

---

## 📚 Example Use Cases

### Use Case 1: Study PA Saturation Impact

```matlab
% Create array of IBO values
IBO_values = [1, 2, 3, 4, 5, 6];

figure;
hold on;

for IBO = IBO_values
    % Configure PA
    config.pa_saturation.enabled = true;
    config.pa_saturation.model = 'rapp';
    config.pa_saturation.IBO_dB = IBO;
    IBO_lin = 10^(IBO/10);
    config.pa_saturation.G = 1;
    config.pa_saturation.Asat = sqrt(IBO_lin);
    config.pa_saturation.p = 2;

    % Disable other impairments
    config.iq_imbalance_tx.enabled = false;
    config.dac_quantization.enabled = false;

    % Test signal
    r_in = linspace(0, 2, 1000);
    x_test = r_in;

    y_out = pa_models(x_test, 'rapp', config.pa_saturation);

    plot(r_in, abs(y_out), 'LineWidth', 2, 'DisplayName', sprintf('IBO=%ddB', IBO));
end

plot(r_in, r_in, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Linear');
xlabel('Input Amplitude');
ylabel('Output Amplitude');
title('PA AM/AM Characteristic vs IBO');
legend('Location', 'best');
grid on;
```

### Use Case 2: Compare PA Models

```matlab
models = {'rapp', 'saleh', 'soft_limiter'};
IBO = 3;

figure;
hold on;

for i = 1:length(models)
    % Configure
    config.pa_saturation.model = models{i};
    config.pa_saturation.IBO_dB = IBO;
    IBO_lin = 10^(IBO/10);

    if strcmp(models{i}, 'rapp')
        config.pa_saturation.G = 1;
        config.pa_saturation.Asat = sqrt(IBO_lin);
        config.pa_saturation.p = 2;
    elseif strcmp(models{i}, 'saleh')
        config.pa_saturation.alpha_a = 2.0;
        config.pa_saturation.beta_a = 1.0/IBO_lin;
        config.pa_saturation.alpha_p = pi/3;
        config.pa_saturation.beta_p = 1.0/IBO_lin;
    else % soft_limiter
        config.pa_saturation.A_lin = sqrt(IBO_lin)*0.7;
        config.pa_saturation.A_sat = sqrt(IBO_lin);
        config.pa_saturation.compress = 0.1;
    end

    % Test
    r_in = linspace(0, 2, 1000);
    y_out = pa_models(r_in, models{i}, config.pa_saturation);

    plot(r_in, abs(y_out), 'LineWidth', 2, 'DisplayName', upper(models{i}));
end

plot(r_in, r_in, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Linear');
xlabel('Input Amplitude');
ylabel('Output Amplitude');
title(sprintf('PA Model Comparison (IBO=%ddB)', IBO));
legend('Location', 'best');
grid on;
```

### Use Case 3: Test All Impairments

```matlab
% Generate test signal
N = 1000;
signal = exp(1j*2*pi*0.05*(1:N)') + 0.1*(randn(N,1) + 1j*randn(N,1));

% Enable all impairments
config.pa_saturation.enabled = true;
config.pa_saturation.model = 'rapp';
config.pa_saturation.IBO_dB = 3;
config.pa_saturation.G = 1;
config.pa_saturation.Asat = sqrt(10^(3/10));
config.pa_saturation.p = 2;
config.pa_saturation.memory_effects = false;

config.iq_imbalance_tx.enabled = true;
config.iq_imbalance_tx.amp_dB = 0.5;
config.iq_imbalance_tx.phase_deg = 5;

config.dac_quantization.enabled = true;
config.dac_quantization.n_bits = 6;
config.dac_quantization.full_scale = 1.0;

config.cfo.enabled = true;
config.cfo.cfo_hz = 100;
config.cfo.fs = 100e3;

config.phase_noise.enabled = true;
config.phase_noise.psd_dBc_Hz = -70;
config.phase_noise.fs = 100e3;

config.iq_imbalance_rx.enabled = true;
config.iq_imbalance_rx.amp_dB = 0.3;
config.iq_imbalance_rx.phase_deg = 3;

config.adc_quantization.enabled = true;
config.adc_quantization.n_bits = 6;
config.adc_quantization.full_scale = 1.0;

% Apply impairments
tx_sig = impairments(signal, config, 'tx');
rx_sig = impairments(tx_sig, config, 'rx');

% Visualize
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
plot(real(signal), imag(signal), 'b.', 'MarkerSize', 4);
grid on; axis equal;
xlabel('In-Phase'); ylabel('Quadrature');
title('Original Signal');

subplot(1,3,2);
plot(real(tx_sig), imag(tx_sig), 'r.', 'MarkerSize', 4);
grid on; axis equal;
xlabel('In-Phase'); ylabel('Quadrature');
title('After TX Impairments');

subplot(1,3,3);
plot(real(rx_sig), imag(rx_sig), 'g.', 'MarkerSize', 4);
grid on; axis equal;
xlabel('In-Phase'); ylabel('Quadrature');
title('After RX Impairments');

sgtitle('Signal Degradation from Impairments');
```

---

## 🐛 Troubleshooting

### Problem: "Undefined function or variable 'pa_models'"
**Solution:**
```matlab
% Make sure you're in the correct directory
cd matlab/pa_saturation

% Or add to path
addpath('matlab/pa_saturation')
```

### Problem: GUI doesn't open
**Solution:**
```matlab
% Check MATLAB version
version  % Should be R2018b or later

% Try running directly
ftn_sim_gui

% If error, check for syntax errors
edit ftn_sim_gui
```

### Problem: Simulation runs but no plots
**Solution:**
```matlab
% Check if figure windows are hidden
shg  % Show graphics

% Or manually create figure
figure;
% ... run plotting code ...
```

### Problem: "Out of memory"
**Solution:**
```matlab
% Reduce problem size
config.n_train = 10000;  % Instead of 100000
config.n_test = 5000;    % Instead of 50000
```

---

## 💡 Tips and Tricks

### Tip 1: Save and Load Configurations
```matlab
% Save config
save('my_config.mat', 'config');

% Load config
load('my_config.mat');
```

### Tip 2: Batch Simulations
```matlab
% Run multiple configurations
IBO_values = [2, 3, 4, 5];

for IBO = IBO_values
    config.pa_saturation.IBO_dB = IBO;
    config.pa_saturation.Asat = sqrt(10^(IBO/10));

    % Run simulation
    % ... your simulation code ...

    % Save results
    filename = sprintf('results_IBO_%d.mat', IBO);
    save(filename, 'results');
end
```

### Tip 3: Quick Visualization
```matlab
% Test PA model quickly
r = linspace(0, 2, 100);
config.pa_saturation.model = 'rapp';
config.pa_saturation.IBO_dB = 3;
config.pa_saturation.G = 1;
config.pa_saturation.Asat = sqrt(10^(3/10));
config.pa_saturation.p = 2;

y = pa_models(r, 'rapp', config.pa_saturation);
plot(r, abs(y)); grid on;
```

---

## 📖 File Reference

| File | Purpose |
|------|---------|
| `pa_models.m` | PA saturation models (Rapp, Saleh, Soft Limiter) |
| `impairments.m` | All hardware impairments (IQ, quantization, phase noise, CFO) |
| `ftn_sim_gui.m` | Interactive GUI for easy configuration |
| `ftn_with_pa_saturation.m` | Original simulation script |
| `HOW_TO_USE.md` | This guide |
| `README.md` | Project overview and theory |

---

## 🎓 For Research

### Systematic Parameter Sweep
```matlab
% Example: Study PA saturation vs IQ imbalance interaction

IBO_range = [2, 3, 4, 5];
IQ_amp_range = [0, 0.5, 1.0, 1.5];

results = zeros(length(IBO_range), length(IQ_amp_range));

for i = 1:length(IBO_range)
    for j = 1:length(IQ_amp_range)
        % Configure
        config.pa_saturation.enabled = true;
        config.pa_saturation.IBO_dB = IBO_range(i);
        config.iq_imbalance_tx.enabled = (IQ_amp_range(j) > 0);
        config.iq_imbalance_tx.amp_dB = IQ_amp_range(j);

        % Run simulation (simplified)
        % results(i,j) = run_and_get_ber(config);

        fprintf('IBO=%d, IQ=%.1f\n', IBO_range(i), IQ_amp_range(j));
    end
end

% Visualize results
figure;
imagesc(IQ_amp_range, IBO_range, results);
colorbar;
xlabel('IQ Imbalance (dB)');
ylabel('IBO (dB)');
title('BER vs PA Saturation and IQ Imbalance');
```

---

## ⚡ Quick Reference Card

```matlab
%% Most Common Commands

% 1. Launch GUI
ftn_sim_gui

% 2. Run baseline simulation
ftn_with_pa_saturation

% 3. Test PA model
r = 0:0.01:2;
cfg.model = 'rapp'; cfg.IBO_dB = 3;
cfg.G = 1; cfg.Asat = sqrt(10^(3/10)); cfg.p = 2;
y = pa_models(r, 'rapp', cfg);
plot(r, abs(y)); grid on;

% 4. Test impairments on signal
signal = randn(1000,1) + 1j*randn(1000,1);
config.pa_saturation.enabled = true;
% ... set other config fields ...
tx_sig = impairments(signal, config, 'tx');
```

---

**Happy Simulating! 🚀**

For detailed theory and background, see README.md in this folder.
