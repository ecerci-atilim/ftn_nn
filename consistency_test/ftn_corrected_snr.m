%% FTN Test - Using MATLAB awgn() for Correct SNR
% En güvenilir SNR: awgn(tx, SNR_dB, 'measured')
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== PARAMETERS ==========
tau = 0.7;
sps = 10;
span = 6;
beta = 0.3;

SNR_range = 0:2:14;
N_test = 1000;
min_errors = 100;
max_bits = 1e6;

%% ========== FILTER SETUP ==========
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;

% ISI
hh = conv(h, h);
[~, peak_idx] = max(hh);
g = hh(peak_idx : step : end);
g = g / g(1);
g = g(abs(g) > 0.001);

fprintf('=== FTN Test with MATLAB awgn() ===\n');
fprintf('τ = %.2f, ISI taps = %d\n\n', tau, length(g));

%% ========== SNR VERIFICATION ==========
fprintf('--- SNR Verification ---\n');

N = 10000;
bits = randi([0 1], N, 1);
symbols = 2 * bits - 1;
tx = conv(upsample(symbols, step), h);

signal_power = mean(tx.^2);
fprintf('Signal power: %.4f\n\n', signal_power);

for SNR_test = [0, 5, 10]
    rx = awgn(tx, SNR_test, 'measured');
    noise_actual = rx - tx;
    noise_var_actual = var(noise_actual);
    snr_actual = 10*log10(signal_power / noise_var_actual);
    fprintf('Target SNR=%2d dB → Actual SNR=%.2f dB (noise_var=%.4f)\n', ...
        SNR_test, snr_actual, noise_var_actual);
end

fprintf('\n');

%% ========== RESULTS STORAGE ==========
n_snr = length(SNR_range);
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));
ber_threshold = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);

%% ========== MAIN TEST ==========
fprintf('--- BER Test ---\n');

for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    
    errors_th = 0;
    errors_sss = 0;
    total = 0;
    
    while (errors_th < min_errors || errors_sss < min_errors) && total < max_bits
        % Generate data
        bits = randi([0 1], N_test, 1);
        symbols = 2 * bits - 1;
        
        % Transmit
        tx = conv(upsample(symbols, step), h);
        
        % ADD NOISE WITH MATLAB awgn - CORRECT SNR
        rx = awgn(tx, SNR_dB, 'measured');
        
        % Matched filter
        rx_mf = conv(rx, h);
        
        % Sample at symbol instants
        rx_sampled = zeros(1, N_test);
        for i = 1:N_test
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
        
        total = total + N_test;
    end
    
    ber_threshold(s_idx) = errors_th / total;
    ber_sss(s_idx) = errors_sss / total;
    
    fprintf('SNR=%2d dB: Threshold=%.2e, SSSgbKSE=%.2e\n', ...
        SNR_dB, ber_threshold(s_idx), ber_sss(s_idx));
end

%% ========== PLOT ==========
figure('Position', [100 100 900 600]);

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Threshold (Uncoded FTN)');
semilogy(SNR_range, ber_sss, 'g-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SSSgbKSE');

grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 11);
title(sprintf('Classical Detectors (τ=%.1f) - Using awgn()', tau), 'FontSize', 13);
ylim([1e-6 1]);

if ~exist('figures', 'dir'), mkdir('figures'); end
% saveas(gcf, 'figures/classical_awgn.png');

%% ========== ANALYSIS ==========
fprintf('\n=== Analysis @ 10 dB ===\n');
idx10 = find(SNR_range == 10);
if ~isempty(idx10)
    fprintf('AWGN Theory:  %.2e\n', ber_theory(idx10));
    fprintf('Threshold:    %.2e\n', ber_threshold(idx10));
    fprintf('SSSgbKSE:     %.2e\n', ber_sss(idx10));
    
    if ber_sss(idx10) > 0
        fprintf('\nSSSgbKSE gain over Threshold: %.1fx\n', ber_threshold(idx10)/ber_sss(idx10));
        fprintf('SSSgbKSE vs AWGN Theory: %.1fx worse\n', ber_sss(idx10)/ber_theory(idx10));
    end
end

%% ========== IMPORTANT NOTE ==========
fprintf('\n========================================\n');
fprintf('NOTE: awgn() uses SNR definition:\n');
fprintf('  SNR = 10*log10(signal_power / noise_power)\n');
fprintf('This is different from Eb/N0!\n');
fprintf('For BPSK: Eb/N0 ≈ SNR (approximately)\n');
fprintf('========================================\n');

%% ========== SAVE ==========
save('mat/comparison/classical_awgn.mat', ...
    'SNR_range', 'ber_theory', 'ber_threshold', 'ber_sss', 'tau', 'N_test');
fprintf('\nSaved: mat/comparison/classical_awgn.mat\n');

%% ========== FUNCTION ==========
function mesSE = SSSgbKSE(mesin, ISI, gbK)
    s = length(mesin);
    L = length(ISI);
    ssd = zeros(1, s);
    mesSE = zeros(1, s);
    
    for n = 1:s
        if n == 1
            ssd(n) = mesin(n);
        else
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
        
        if n > gbK
            for i = (n - gbK):n
                backward_taps = min(i-1, L-1);
                forward_taps = min(n-i, L-1);
                
                val = mesin(i);
                
                if forward_taps > 0 && i < n
                    for k = 1:forward_taps
                        if k+1 <= L && i+k <= n
                            val = val - sign(ssd(i+k)) * ISI(k+1);
                        end
                    end
                end
                
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
    
    mesSE(mesSE == 0) = ssd(mesSE == 0);
end