function signal_out = impairments(signal_in, config, stage)
% IMPAIRMENTS - Apply hardware impairments to signal
%
% signal_out = impairments(signal_in, config, stage)
%
% Inputs:
%   signal_in - Input signal (complex baseband)
%   config    - Configuration structure with impairment settings
%   stage     - 'tx' or 'rx' (which stage to apply)
%
% Output:
%   signal_out - Signal after impairments
%
% Configuration Structure:
%   config.pa_saturation.enabled = true/false
%   config.pa_saturation.model = 'rapp'/'saleh'/'soft_limiter'
%   config.pa_saturation.IBO_dB = 3
%   config.pa_saturation.memory_effects = true/false
%
%   config.iq_imbalance_tx.enabled = true/false
%   config.iq_imbalance_tx.amp_dB = 0.5
%   config.iq_imbalance_tx.phase_deg = 5
%
%   config.dac_quantization.enabled = true/false
%   config.dac_quantization.n_bits = 8
%
%   config.cfo.enabled = true/false
%   config.cfo.cfo_hz = 100
%   config.cfo.fs = 1e6
%
%   config.phase_noise.enabled = true/false
%   config.phase_noise.psd_dBc_Hz = -80
%   config.phase_noise.fs = 1e6
%
%   config.iq_imbalance_rx.enabled = true/false
%   config.iq_imbalance_rx.amp_dB = 0.3
%   config.iq_imbalance_rx.phase_deg = 3
%
%   config.adc_quantization.enabled = true/false
%   config.adc_quantization.n_bits = 8
%
% Author: Emre Cerci
% Date: January 2026

    signal_out = signal_in;

    if strcmpi(stage, 'tx')
        % Transmitter chain: IQ -> DAC -> PA

        % 1. TX IQ Imbalance
        if isfield(config, 'iq_imbalance_tx') && config.iq_imbalance_tx.enabled
            signal_out = apply_iq_imbalance(signal_out, config.iq_imbalance_tx);
        end

        % 2. DAC Quantization
        if isfield(config, 'dac_quantization') && config.dac_quantization.enabled
            signal_out = apply_quantization(signal_out, config.dac_quantization);
        end

        % 3. PA Saturation
        if isfield(config, 'pa_saturation') && config.pa_saturation.enabled
            if isfield(config.pa_saturation, 'memory_effects') && config.pa_saturation.memory_effects
                signal_out = apply_pa_with_memory(signal_out, config.pa_saturation);
            else
                signal_out = pa_models(signal_out, config.pa_saturation.model, config.pa_saturation);
            end
        end

    elseif strcmpi(stage, 'rx')
        % Receiver chain: CFO -> Phase Noise -> RX IQ -> ADC

        % 1. Carrier Frequency Offset
        if isfield(config, 'cfo') && config.cfo.enabled
            signal_out = apply_cfo(signal_out, config.cfo);
        end

        % 2. Phase Noise
        if isfield(config, 'phase_noise') && config.phase_noise.enabled
            signal_out = apply_phase_noise(signal_out, config.phase_noise);
        end

        % 3. RX IQ Imbalance
        if isfield(config, 'iq_imbalance_rx') && config.iq_imbalance_rx.enabled
            signal_out = apply_iq_imbalance(signal_out, config.iq_imbalance_rx);
        end

        % 4. ADC Quantization
        if isfield(config, 'adc_quantization') && config.adc_quantization.enabled
            signal_out = apply_quantization(signal_out, config.adc_quantization);
        end
    end
end


%% Helper Functions

function y = apply_iq_imbalance(x, cfg)
    % Apply IQ imbalance
    % y = (1+α)·I + j·(1-α)·Q·exp(jφ)

    amp_dB = cfg.amp_dB;
    phase_deg = cfg.phase_deg;

    % Convert to linear
    alpha = (10^(amp_dB/20) - 1) / 2;
    phi = deg2rad(phase_deg);

    I = real(x);
    Q = imag(x);

    % Apply imbalance
    I_imb = (1 + alpha) * I;
    Q_imb = (1 - alpha) * Q;

    % Phase rotation
    y = I_imb + 1j * Q_imb * exp(1j * phi);
end


function y = apply_quantization(x, cfg)
    % Apply ADC/DAC quantization

    n_bits = cfg.n_bits;

    if isfield(cfg, 'full_scale')
        full_scale = cfg.full_scale;
    else
        full_scale = max(abs([real(x); imag(x)]));
    end

    % Quantization levels
    n_levels = 2^n_bits;
    step_size = 2 * full_scale / n_levels;

    % Quantize I and Q separately
    I = real(x);
    Q = imag(x);

    I_quant = round(I / step_size) * step_size;
    Q_quant = round(Q / step_size) * step_size;

    % Clip to range
    I_quant = max(min(I_quant, full_scale), -full_scale);
    Q_quant = max(min(Q_quant, full_scale), -full_scale);

    y = I_quant + 1j * Q_quant;
end


function y = apply_cfo(x, cfg)
    % Apply Carrier Frequency Offset
    % y[n] = x[n] · exp(j·2π·Δf·n/fs)

    cfo_hz = cfg.cfo_hz;
    fs = cfg.fs;

    n = (0:length(x)-1)';
    phase_shift = exp(1j * 2 * pi * cfo_hz * n / fs);

    y = x .* phase_shift;
end


function y = apply_phase_noise(x, cfg)
    % Apply Phase Noise (Wiener process model)
    % φ[n] = φ[n-1] + Δφ[n], where Δφ[n] ~ N(0, σ²)

    psd_dBc_Hz = cfg.psd_dBc_Hz;
    fs = cfg.fs;

    % Convert PSD to variance (simplified)
    f_3dB = 10^(psd_dBc_Hz / 10);
    sigma_phi = sqrt(2 * pi * f_3dB / fs);

    % Generate phase noise (Wiener process)
    phase_noise = cumsum(randn(size(x)) * sigma_phi);

    % Apply to signal
    y = x .* exp(1j * phase_noise);
end


function y = apply_pa_with_memory(x, cfg)
    % PA with memory effects (simplified Memory Polynomial)
    % y[n] = PA(x[n]) + Σ αk · PA(x[n-k])

    depth = 3;
    if isfield(cfg, 'memory_depth')
        depth = cfg.memory_depth;
    end

    % Memoryless component
    y = pa_models(x, cfg.model, cfg);

    % Memory components
    memory_coeffs = [0.05, 0.03, 0.01];
    memory_coeffs = memory_coeffs(1:min(depth, length(memory_coeffs)));

    for k = 1:length(memory_coeffs)
        x_delayed = [zeros(k, 1); x(1:end-k)];
        y = y + memory_coeffs(k) * pa_models(x_delayed, cfg.model, cfg);
    end
end


%% Configuration Helper Functions

function config = get_default_config()
    % Return default configuration with all impairments disabled

    config.pa_saturation.enabled = false;
    config.pa_saturation.model = 'rapp';
    config.pa_saturation.IBO_dB = 3;
    config.pa_saturation.memory_effects = false;

    config.iq_imbalance_tx.enabled = false;
    config.iq_imbalance_tx.amp_dB = 0.5;
    config.iq_imbalance_tx.phase_deg = 5;

    config.dac_quantization.enabled = false;
    config.dac_quantization.n_bits = 8;
    config.dac_quantization.full_scale = 1.0;

    config.cfo.enabled = false;
    config.cfo.cfo_hz = 100;
    config.cfo.fs = 1e6;

    config.phase_noise.enabled = false;
    config.phase_noise.psd_dBc_Hz = -80;
    config.phase_noise.fs = 1e6;

    config.iq_imbalance_rx.enabled = false;
    config.iq_imbalance_rx.amp_dB = 0.3;
    config.iq_imbalance_rx.phase_deg = 3;

    config.adc_quantization.enabled = false;
    config.adc_quantization.n_bits = 8;
    config.adc_quantization.full_scale = 1.0;
end


function print_config(config)
    % Print current configuration

    fprintf('============================================================\n');
    fprintf('Impairment Configuration\n');
    fprintf('============================================================\n\n');

    fprintf('[Transmitter Impairments]\n');

    if isfield(config, 'iq_imbalance_tx') && config.iq_imbalance_tx.enabled
        fprintf('  ✓ TX IQ Imbalance: amp=%.1fdB, phase=%.1f°\n', ...
            config.iq_imbalance_tx.amp_dB, config.iq_imbalance_tx.phase_deg);
    else
        fprintf('  ✗ TX IQ Imbalance: OFF\n');
    end

    if isfield(config, 'dac_quantization') && config.dac_quantization.enabled
        fprintf('  ✓ DAC Quantization: %d bits\n', config.dac_quantization.n_bits);
    else
        fprintf('  ✗ DAC Quantization: OFF\n');
    end

    if isfield(config, 'pa_saturation') && config.pa_saturation.enabled
        mem_str = '';
        if isfield(config.pa_saturation, 'memory_effects') && config.pa_saturation.memory_effects
            mem_str = ' (with memory)';
        end
        fprintf('  ✓ PA Saturation: %s, IBO=%ddB%s\n', ...
            config.pa_saturation.model, config.pa_saturation.IBO_dB, mem_str);
    else
        fprintf('  ✗ PA Saturation: OFF\n');
    end

    fprintf('\n[Receiver Impairments]\n');

    if isfield(config, 'cfo') && config.cfo.enabled
        fprintf('  ✓ CFO: %.1f Hz\n', config.cfo.cfo_hz);
    else
        fprintf('  ✗ CFO: OFF\n');
    end

    if isfield(config, 'phase_noise') && config.phase_noise.enabled
        fprintf('  ✓ Phase Noise: %.1f dBc/Hz\n', config.phase_noise.psd_dBc_Hz);
    else
        fprintf('  ✗ Phase Noise: OFF\n');
    end

    if isfield(config, 'iq_imbalance_rx') && config.iq_imbalance_rx.enabled
        fprintf('  ✓ RX IQ Imbalance: amp=%.1fdB, phase=%.1f°\n', ...
            config.iq_imbalance_rx.amp_dB, config.iq_imbalance_rx.phase_deg);
    else
        fprintf('  ✗ RX IQ Imbalance: OFF\n');
    end

    if isfield(config, 'adc_quantization') && config.adc_quantization.enabled
        fprintf('  ✓ ADC Quantization: %d bits\n', config.adc_quantization.n_bits);
    else
        fprintf('  ✗ ADC Quantization: OFF\n');
    end

    fprintf('============================================================\n');
end
