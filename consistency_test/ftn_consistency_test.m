%% FTN Consistency Analysis Framework
% Bu kod tüm detektorları aynı koşullarda test eder
% Power normalization, SNR tanımı, N etkisi analizi
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== CRITICAL PARAMETERS ==========
% Tüm testlerde bu parametreler sabit kalacak

% System
sps = 10;           % samples per symbol
span = 6;           % filter span
beta = 0.3;         % roll-off factor
tau = 0.7;          % FTN compression factor

% SNR
SNR_range = 0:2:14;

% Monte Carlo
min_errors = 100;   % minimum errors to collect
max_bits = 1e6;     % maximum bits to test

% NN parameters (from training)
window_len = 31;
num_feedback = 4;

%% ========== FILTER SETUP ==========
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);  % energy normalize
step = round(sps * tau);
delay = span * sps;

% ISI sequence (for classical detectors)
hh = conv(h, h);  % matched filter response
[~, peak_idx] = max(hh);
g = hh(peak_idx : step : end);
g = g / g(1);  % normalize so g(0) = 1
g = g(abs(g) > 0.001);  % truncate negligible taps
ISI_length = length(g);

fprintf('=== FTN Consistency Test ===\n');
fprintf('τ = %.2f, ISI taps = %d\n\n', tau, ISI_length);

%% ========== TEST 1: Power Normalization Effect ==========
fprintf('--- TEST 1: Power Normalization Effect ---\n');

N = 10000;
SNR_test = 10;  % dB

% Generate signal
bits = randi([0 1], N, 1);
symbols = 2 * bits - 1;

tx = conv(upsample(symbols, step), h);
EbN0_linear = 10^(SNR_test/10);
noise_var = 1 / (2 * EbN0_linear);
rx = tx + sqrt(noise_var) * randn(size(tx));
rx_mf = conv(rx, h);

% Statistics before normalization
fprintf('Before normalization:\n');
fprintf('  rx_mf std = %.4f\n', std(rx_mf));
fprintf('  rx_mf mean = %.4f\n', mean(rx_mf));

% After normalization
rx_mf_norm = rx_mf / std(rx_mf);
fprintf('After normalization:\n');
fprintf('  rx_mf_norm std = %.4f\n', std(rx_mf_norm));

% What this means for noise
fprintf('\nIMPLICATION:\n');
fprintf('  Original noise var = %.4f\n', noise_var);
fprintf('  After norm, effective noise var = %.4f\n', noise_var / var(rx_mf));
fprintf('  SNR shift ≈ %.1f dB\n', 10*log10(var(rx_mf)));

%% ========== TEST 2: N (Sequence Length) Effect ==========
fprintf('\n--- TEST 2: Sequence Length Effect ---\n');

N_values = [20, 50, 100, 500, 1000, 5000, 10000, 50000];
SNR_test = 10;

fprintf('Testing different N values at SNR = %d dB\n', SNR_test);
fprintf('(Using simple threshold detection)\n\n');

ber_vs_N = zeros(size(N_values));

for n_idx = 1:length(N_values)
    N = N_values(n_idx);
    
    errors = 0;
    total = 0;
    
    while errors < min_errors && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        EbN0_linear = 10^(SNR_test/10);
        noise_var = 1 / (2 * EbN0_linear);
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx, h);
        
        % Simple threshold detection
        for i = 1:N
            idx = (i-1) * step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf)
                pred_bit = real(rx_mf(idx)) > 0;
                if pred_bit ~= bits(i)
                    errors = errors + 1;
                end
                total = total + 1;
            end
        end
    end
    
    ber_vs_N(n_idx) = errors / total;
    fprintf('  N = %5d: BER = %.2e\n', N, ber_vs_N(n_idx));
end

% Plot N effect
figure('Position', [100 100 600 400]);
semilogx(N_values, ber_vs_N, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('Sequence Length N');
ylabel('BER');
title(sprintf('Effect of Sequence Length (τ=%.1f, SNR=%d dB)', tau, SNR_test));
% saveas(gcf, 'figures/N_effect_test.png');

%% ========== TEST 3: Consistent Comparison (NO Normalization) ==========
fprintf('\n--- TEST 3: Consistent Comparison (No Power Norm) ---\n');

N = 1000;  % Fixed moderate length

% Results storage
ber_threshold = zeros(1, length(SNR_range));
ber_sss = zeros(1, length(SNR_range));
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

for s_idx = 1:length(SNR_range)
    SNR_dB = SNR_range(s_idx);
    EbN0_linear = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_linear);
    
    errors_th = 0;
    errors_sss = 0;
    total = 0;
    
    while (errors_th < min_errors || errors_sss < min_errors) && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx, h);
        
        % Sample at symbol instants (NO normalization)
        rx_sampled = zeros(1, N);
        for i = 1:N
            idx = (i-1) * step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf)
                rx_sampled(i) = real(rx_mf(idx));
            end
        end
        
        % 1. Threshold detection
        bits_th = rx_sampled > 0;
        errors_th = errors_th + sum(bits_th ~= bits');
        
        % 2. SSSgbKSE
        bits_sss = SSSgbKSE(rx_sampled, g, 4) > 0;
        errors_sss = errors_sss + sum(bits_sss ~= bits');
        
        total = total + N;
    end
    
    ber_threshold(s_idx) = errors_th / total;
    ber_sss(s_idx) = errors_sss / total;
    
    fprintf('SNR = %2d dB: Threshold = %.2e, SSSgbKSE = %.2e\n', ...
        SNR_dB, ber_threshold(s_idx), ber_sss(s_idx));
end

% Plot
figure('Position', [100 100 800 500]);
semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss, 'g-s', 'LineWidth', 1.5, 'DisplayName', 'SSSgbKSE');
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
legend('Location', 'southwest');
title(sprintf('Classical Detectors - No Power Norm (τ=%.1f, N=%d)', tau, N));
ylim([1e-5 1]);
% saveas(gcf, 'figures/classical_consistent.png');

%% ========== TEST 4: SNR Definition Check ==========
fprintf('\n--- TEST 4: SNR Definition Verification ---\n');

N = 10000;
SNR_test = 10;

bits = randi([0 1], N, 1);
symbols = 2 * bits - 1;

% Method 1: Manual noise variance (our method)
tx1 = conv(upsample(symbols, step), h);
EbN0_linear = 10^(SNR_test/10);
noise_var = 1 / (2 * EbN0_linear);
rx1 = tx1 + sqrt(noise_var) * randn(size(tx1));

signal_power = var(tx1);
noise_power = noise_var;
actual_SNR_method1 = 10*log10(signal_power / noise_power);

% Method 2: Using awgn function (MATLAB built-in)
tx2 = conv(upsample(symbols, step), h);
rx2 = awgn(tx2, SNR_test, 'measured');

fprintf('Target SNR: %d dB\n', SNR_test);
fprintf('Method 1 (manual): Actual SNR = %.2f dB\n', actual_SNR_method1);
fprintf('Signal power = %.4f, Noise var = %.4f\n', signal_power, noise_var);

%% ========== FUNCTIONS ==========

function mesSE = SSSgbKSE(mesin, ISI, gbK)
    % Symbol-by-Symbol detection with Go-back-K Sequence Estimation
    % mesin: matched filter output samples
    % ISI: ISI tap coefficients [g(0), g(1), g(2), ...]
    % gbK: go-back length
    
    s = length(mesin);
    L = length(ISI);
    ssd = zeros(1, s);
    mesSE = zeros(1, s);
    
    for n = 1:s
        % --- SSD: Symbol-by-Symbol Detection ---
        if n == 1
            ssd(n) = mesin(n);
        else
            % Backward ISI cancellation
            isi_taps = min(n-1, L-1);
            if isi_taps > 0
                isi_cancel = 0;
                for k = 1:isi_taps
                    if k+1 <= L
                        isi_cancel = isi_cancel + sign(ssd(n-k)) * ISI(k+1);
                    end
                end
                ssd(n) = mesin(n) - isi_cancel;
            else
                ssd(n) = mesin(n);
            end
        end
        
        % --- Go-back K: Refine past decisions ---
        if n > gbK
            for i = (n - gbK):n
                backward_taps = min(i-1, L-1);
                forward_taps = min(n-i, L-1);
                
                val = mesin(i);
                
                % Forward ISI (from future symbols - use ssd)
                if forward_taps > 0 && i < n
                    for k = 1:forward_taps
                        if k+1 <= L && i+k <= n
                            val = val - sign(ssd(i+k)) * ISI(k+1);
                        end
                    end
                end
                
                % Backward ISI (from past symbols - use mesSE)
                if backward_taps > 0 && i > 1
                    for k = 1:backward_taps
                        if k+1 <= L && i-k >= 1
                            val = val - sign(mesSE(i-k)) * ISI(k+1);
                        end
                    end
                end
                
                mesSE(i) = val;
            end
        end
    end
    
    % Fill unprocessed symbols
    mesSE(mesSE == 0) = ssd(mesSE == 0);
end
