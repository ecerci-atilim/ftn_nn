% FTN_WITH_PA_SATURATION - FTN Signaling with Power Amplifier Saturation
%
% Demonstrates the advantage of Fractionally Spaced Equalization (FSE) over
% Symbol-Rate Sampling in the presence of PA nonlinearity.
%
% Key Points:
%   1. PA saturation creates nonlinearity BEFORE the channel
%   2. Channel remains linear (AWGN)
%   3. Fractional sampling at receiver captures nonlinear distortion better
%   4. Symbol-rate sampling loses information about nonlinear effects
%
% System Model:
%   Bits -> BPSK -> FTN (SRRC, tau<1) -> PA Saturation -> AWGN ->
%   -> Matched Filter -> Sampling (Symbol-rate vs Fractional) -> Detection
%
% Author: Emre Cerci
% Date: January 2026

clear; close all; clc;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% FTN Parameters
tau = 0.7;                  % FTN compression factor
beta = 0.3;                 % SRRC roll-off (same as reference)
sps = 10;                   % Samples per symbol
span = 6;                   % Pulse span in symbols

% PA Parameters
PA_MODEL = 'rapp';          % Options: 'rapp', 'saleh', 'soft_limiter'
IBO_dB = 3;                 % Input Back-Off in dB (controls saturation level)

% Simulation Parameters
SNR_dB_range = 0:2:14;      % SNR range for BER curves
N_symbols_train = 100000;   % Training symbols (for NN equalizer)
N_symbols_test = 50000;     % Test symbols per SNR point
N_window = 8;               % Window size (2*N_window + 1 samples)

% Fractional Sampling
L_frac = 2;                 % Fractional sampling factor (L=2 means T/2 spacing)

fprintf('========================================\n');
fprintf('FTN with PA Saturation Simulation\n');
fprintf('========================================\n');
fprintf('tau = %.2f, beta = %.2f, sps = %d\n', tau, beta, sps);
fprintf('PA Model: %s, IBO = %d dB\n', PA_MODEL, IBO_dB);
fprintf('Fractional factor: L = %d (T/%d spacing)\n', L_frac, L_frac);
fprintf('========================================\n\n');

%% ========================================================================
%% PULSE SHAPING FILTER (Square Root Raised Cosine)
%% ========================================================================

% Generate SRRC pulse using rcosdesign (same as reference implementation)
h_srrc = rcosdesign(beta, span, sps, 'sqrt');
h_srrc = h_srrc / norm(h_srrc);  % Normalize to unit energy
delay = span * sps;

%% ========================================================================
%% PA SATURATION CONFIGURATION
%% ========================================================================

% Configure PA parameters based on IBO
IBO_lin = 10^(IBO_dB/10);

switch lower(PA_MODEL)
    case 'rapp'
        pa_params.G = 1;
        pa_params.Asat = sqrt(IBO_lin);  % Asat adjusted by IBO
        pa_params.p = 2;                 % Smoothness (2 = typical SSPA)

    case 'saleh'
        pa_params.alpha_a = 2.0;
        pa_params.beta_a = 1.0 / IBO_lin;
        pa_params.alpha_p = pi/3;
        pa_params.beta_p = 1.0 / IBO_lin;

    case 'soft_limiter'
        pa_params.A_lin = sqrt(IBO_lin) * 0.7;
        pa_params.A_sat = sqrt(IBO_lin);
        pa_params.compress = 0.1;
end

%% ========================================================================
%% TRANSMITTER FUNCTION
%% ========================================================================

function [tx_sig, rx_mf, symbols, bits] = transmit_ftn_with_pa(N_sym, SNR_dB, ...
    tau, sps, h_srrc, delay, PA_MODEL, pa_params)

    % Generate random bits
    bits = randi([0 1], N_sym, 1);
    symbols = 2*bits - 1;  % BPSK

    % FTN: Compress symbol rate
    step = round(tau * sps);

    % Upsample
    tx_up = zeros(N_sym * step, 1);
    tx_up(1:step:end) = symbols;

    % Pulse shaping (SRRC)
    tx_shaped = conv(tx_up, h_srrc, 'full');

    % ===== PA SATURATION (NONLINEARITY) =====
    tx_sig = pa_models(tx_shaped, PA_MODEL, pa_params);

    % AWGN Channel (LINEAR)
    EbN0 = 10^(SNR_dB/10);
    noise_power = 1 / (2 * EbN0);  % For BPSK
    noise = sqrt(noise_power) * randn(size(tx_sig));  % Real noise for BPSK
    rx_noisy = tx_sig + noise;

    % Matched Filter
    rx_mf = conv(rx_noisy, h_srrc, 'full');
    rx_mf = rx_mf / std(rx_mf);  % Normalize
end

%% ========================================================================
%% DETECTION FUNCTIONS
%% ========================================================================

% Symbol-Rate Sampling Detector (Classical)
function [bits_hat, BER] = detect_symbol_rate(rx_mf, bits, tau, sps, delay, N_window)
    step = round(tau * sps);
    total_delay = 2 * delay;
    N_sym = length(bits);

    bits_hat = zeros(N_sym, 1);

    for k = 1:N_sym
        % Sample at symbol rate
        center_idx = total_delay + 1 + (k-1) * step;  % MATLAB 1-based indexing

        if center_idx > 0 && center_idx <= length(rx_mf)
            % Simple threshold detection on symbol-rate sample
            bits_hat(k) = real(rx_mf(center_idx)) > 0;
        end
    end

    BER = mean(bits_hat ~= bits);
end

% Fractional Sampling Detector (with windowed averaging)
function [bits_hat, BER] = detect_fractional(rx_mf, bits, tau, sps, delay, N_window, L_frac)
    step = round(tau * sps);
    frac_step = round(step / L_frac);  % Fractional spacing
    total_delay = 2 * delay;
    N_sym = length(bits);

    bits_hat = zeros(N_sym, 1);

    for k = 1:N_sym
        center_idx = total_delay + 1 + (k-1) * step;  % MATLAB 1-based indexing

        % Extract fractional samples around symbol time
        frac_samples = [];
        for offset = -N_window:N_window
            idx = center_idx + offset * frac_step;
            if idx > 0 && idx <= length(rx_mf)
                frac_samples = [frac_samples; real(rx_mf(idx))];
            end
        end

        % Decision: weighted average or max-likelihood
        % Simple approach: use maximum value (peak detection)
        if ~isempty(frac_samples)
            decision_stat = mean(frac_samples);  % Or: max(frac_samples)
            bits_hat(k) = decision_stat > 0;
        end
    end

    BER = mean(bits_hat ~= bits);
end

%% ========================================================================
%% MAIN SIMULATION LOOP
%% ========================================================================

BER_symbol_rate = zeros(size(SNR_dB_range));
BER_fractional = zeros(size(SNR_dB_range));

fprintf('Running BER simulation...\n');
fprintf('SNR(dB) | Symbol-Rate BER | Fractional BER | Gain(dB)\n');
fprintf('--------|-----------------|----------------|----------\n');

for snr_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(snr_idx);

    % Generate test data
    [~, rx_mf, ~, bits] = transmit_ftn_with_pa(N_symbols_test, SNR_dB, ...
        tau, sps, h_srrc, delay, PA_MODEL, pa_params);

    % Symbol-rate detection
    [~, BER_sym] = detect_symbol_rate(rx_mf, bits, tau, sps, delay, N_window);
    BER_symbol_rate(snr_idx) = BER_sym;

    % Fractional sampling detection
    [~, BER_frac] = detect_fractional(rx_mf, bits, tau, sps, delay, N_window, L_frac);
    BER_fractional(snr_idx) = BER_frac;

    % Calculate gain
    if BER_frac > 0 && BER_sym > 0
        gain_dB = 10*log10(BER_sym / BER_frac);
    else
        gain_dB = NaN;
    end

    fprintf('  %2d    |   %.2e      |   %.2e     |  %.2f dB\n', ...
        SNR_dB, BER_sym, BER_frac, gain_dB);
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

figure('Position', [100 100 1200 800]);

% Subplot 1: PA Characteristic
subplot(2,2,1);
r_in = linspace(0, 2, 1000);
x_test = r_in;
y_test = pa_models(x_test, PA_MODEL, pa_params);
plot(r_in, abs(y_test), 'b-', 'LineWidth', 2);
hold on;
plot(r_in, r_in, 'r--', 'LineWidth', 1.5);
grid on;
xlabel('Input Amplitude');
ylabel('Output Amplitude');
title(sprintf('PA Characteristic (%s, IBO=%ddB)', PA_MODEL, IBO_dB));
legend('PA Output', 'Linear', 'Location', 'best');

% Subplot 2: Constellation Diagrams
subplot(2,2,2);
[tx_sig_demo, ~, ~, ~] = transmit_ftn_with_pa(1000, 10, tau, sps, h_srrc, delay, PA_MODEL, pa_params);
plot(real(tx_sig_demo), imag(tx_sig_demo), 'b.', 'MarkerSize', 4);
grid on; axis equal;
xlabel('In-Phase'); ylabel('Quadrature');
title('Transmitted Signal Constellation (After PA)');

% Subplot 3: BER Curves
subplot(2,2,3);
semilogy(SNR_dB_range, BER_symbol_rate, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Symbol-Rate Sampling');
hold on;
semilogy(SNR_dB_range, BER_fractional, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('Fractional (L=%d)', L_frac));
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title(sprintf('BER Performance (\\tau=%.2f, PA=%s)', tau, PA_MODEL));
legend('Location', 'best');
ylim([1e-5 1]);

% Subplot 4: Performance Gain
subplot(2,2,4);
valid_idx = (BER_symbol_rate > 0) & (BER_fractional > 0);
gain_dB = 10*log10(BER_symbol_rate(valid_idx) ./ BER_fractional(valid_idx));
plot(SNR_dB_range(valid_idx), gain_dB, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Performance Gain (dB)');
title('Fractional Sampling Advantage');

sgtitle('FTN with PA Saturation: Fractional vs Symbol-Rate Sampling');

% Save figure
saveas(gcf, 'matlab_ftn_pa_saturation_results.png');

fprintf('\n========================================\n');
fprintf('Simulation Complete!\n');
fprintf('Figure saved: matlab_ftn_pa_saturation_results.png\n');
fprintf('========================================\n');
