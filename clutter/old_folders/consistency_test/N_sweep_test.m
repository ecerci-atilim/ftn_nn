%% N Sweep Test - Sequence Length Effect Analysis
% Hocanın sorusu: "N küçülürse performans artar mı?"
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== PARAMETERS ==========
tau = 0.7;
SNR_dB = 10;  % Fixed SNR for N analysis
N_values = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000];
min_errors = 50;  % Reduced for faster testing
max_bits = 5e5;

USE_POWER_NORM = false;

%% ========== SETUP ==========
sps = 10; span = 6;
h = rcosdesign(0.3, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;

% ISI
hh = conv(h, h);
[~, peak_idx] = max(hh);
g = hh(peak_idx : step : end);
g = g / g(1);
g = g(abs(g) > 0.001);

EbN0_linear = 10^(SNR_dB/10);
noise_var = 1 / (2 * EbN0_linear);

fprintf('=== N Sweep Test ===\n');
fprintf('τ = %.1f, SNR = %d dB\n', tau, SNR_dB);
fprintf('ISI taps: %d\n\n', length(g));

%% ========== RESULTS STORAGE ==========
n_N = length(N_values);
ber_threshold = zeros(1, n_N);
ber_sss = zeros(1, n_N);

%% ========== N SWEEP TEST ==========
for n_idx = 1:n_N
    N = N_values(n_idx);
    
    errors_th = 0;
    errors_sss = 0;
    total = 0;
    
    while (errors_th < min_errors || errors_sss < min_errors) && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx, h);
        
        if USE_POWER_NORM
            rx_mf = rx_mf / std(rx_mf);
        end
        
        % Sample at symbol instants
        rx_sampled = zeros(1, N);
        for i = 1:N
            idx = (i-1) * step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf)
                rx_sampled(i) = real(rx_mf(idx));
            end
        end
        
        % Threshold detection
        bits_th = rx_sampled > 0;
        errors_th = errors_th + sum(bits_th ~= bits');
        
        % SSSgbKSE
        bits_sss = SSSgbKSE(rx_sampled, g, 4) > 0;
        errors_sss = errors_sss + sum(bits_sss ~= bits');
        
        total = total + N;
    end
    
    ber_threshold(n_idx) = errors_th / total;
    ber_sss(n_idx) = errors_sss / total;
    
    fprintf('N = %5d: Threshold = %.2e, SSSgbKSE = %.2e\n', N, ber_threshold(n_idx), ber_sss(n_idx));
end

%% ========== PLOT ==========
figure('Position', [100 100 800 500]);

semilogx(N_values, ber_threshold, 'r-^', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Threshold');
hold on;
semilogx(N_values, ber_sss, 'g-s', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SSSgbKSE');

% AWGN reference line
ber_awgn = qfunc(sqrt(2 * EbN0_linear));
yline(ber_awgn, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');

grid on;
xlabel('Sequence Length N', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 11);
title(sprintf('Effect of Sequence Length (τ=%.1f, SNR=%d dB)', tau, SNR_dB));

%% ========== SAVE ==========
if ~exist('figures', 'dir'), mkdir('figures'); end
% saveas(gcf, sprintf('figures/N_sweep_tau%02d_SNR%d.png', tau*10, SNR_dB));

if ~exist('mat/comparison', 'dir'), mkdir('mat/comparison'); end
save(sprintf('mat/comparison/N_sweep_tau%02d_SNR%d.mat', tau*10, SNR_dB), ...
    'N_values', 'ber_threshold', 'ber_sss', 'tau', 'SNR_dB');

%% ========== ANALYSIS ==========
fprintf('\n=== Analysis ===\n');
fprintf('BER change from N=20 to N=50000:\n');
fprintf('  Threshold: %.2e → %.2e (%.1fx)\n', ber_threshold(1), ber_threshold(end), ber_threshold(end)/ber_threshold(1));
fprintf('  SSSgbKSE:  %.2e → %.2e (%.1fx)\n', ber_sss(1), ber_sss(end), ber_sss(end)/ber_sss(1));

% ISI memory analysis
isi_memory = 2 * span / tau;
fprintf('\nISI memory ≈ %.0f symbols\n', isi_memory);
fprintf('Pattern space ≈ 2^%.0f = %.0e\n', isi_memory, 2^isi_memory);

fprintf('\n=== Interpretation ===\n');
if ber_threshold(1) < ber_threshold(end) * 0.8
    fprintf('WARNING: N has significant effect!\n');
    fprintf('  Small N → better BER (possible edge effects)\n');
else
    fprintf('N effect is minimal for classical detectors.\n');
end

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
