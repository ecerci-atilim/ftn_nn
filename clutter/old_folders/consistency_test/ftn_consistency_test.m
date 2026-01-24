%% FTN Sequence Length Effect - Rigorous Test
% Çoklu bağımsız çalıştırma ile güvenilir istatistik
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% Parameters
tau = 0.7;
SNR_dB = 10;
N_values = [50, 100, 500, 1000, 5000, 10000, 50000];

n_trials = 10;          % Her N için 10 bağımsız deneme
min_errors = 500;       % Daha fazla hata
max_bits = 5e6;         % Daha yüksek limit

sps = 10; span = 6;
h = rcosdesign(0.3, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;

% ISI
hh = conv(h, h);
[~, pk] = max(hh);
g = hh(pk:step:end); g = g/g(1); g = g(abs(g)>0.001);

EbN0_lin = 10^(SNR_dB/10);
noise_var = 1 / (2 * EbN0_lin);

fprintf('=== Rigorous N Effect Test (τ=%.1f, SNR=%d dB) ===\n', tau, SNR_dB);
fprintf('Trials per N: %d, Min errors: %d\n\n', n_trials, min_errors);

%% Results storage
ber_th_all = zeros(length(N_values), n_trials);
ber_sss_all = zeros(length(N_values), n_trials);

%% Main Loop
for n_idx = 1:length(N_values)
    N = N_values(n_idx);
    fprintf('N = %5d: ', N);
    
    for trial = 1:n_trials
        errors_th = 0;
        errors_sss = 0;
        total_bits = 0;
        
        while (errors_th < min_errors || errors_sss < min_errors) && total_bits < max_bits
            bits = randi([0 1], N, 1);
            symbols = 2*bits - 1;
            
            tx = conv(upsample(symbols, step), h);
            rx = tx + sqrt(noise_var) * randn(size(tx));
            rx_mf = conv(rx, h);
            
            rx_sym = zeros(1, N);
            for i = 1:N
                idx = (i-1)*step + 1 + delay;
                if idx > 0 && idx <= length(rx_mf)
                    rx_sym(i) = rx_mf(idx);
                end
            end
            
            % Threshold
            det_th = rx_sym > 0;
            errors_th = errors_th + sum(det_th ~= bits');
            
            % SSSgbKSE  
            det_sss = SSSgbKSE(rx_sym, g, 4) > 0;
            errors_sss = errors_sss + sum(det_sss ~= bits');
            
            total_bits = total_bits + N;
        end
        
        ber_th_all(n_idx, trial) = errors_th / total_bits;
        ber_sss_all(n_idx, trial) = errors_sss / total_bits;
        fprintf('.');
    end
    
    % Statistics
    th_mean = mean(ber_th_all(n_idx,:));
    th_std = std(ber_th_all(n_idx,:));
    sss_mean = mean(ber_sss_all(n_idx,:));
    sss_std = std(ber_sss_all(n_idx,:));
    
    fprintf(' Th=%.3f±%.3f%%, SSS=%.4f±%.4f%%\n', ...
        th_mean*100, th_std*100, sss_mean*100, sss_std*100);
end

%% Statistical Analysis
fprintf('\n=== Statistical Analysis ===\n');

th_means = mean(ber_th_all, 2);
sss_means = mean(ber_sss_all, 2);

% Correlation with N (using base MATLAB)
R_th = corrcoef(log10(N_values)', th_means);
r_th = R_th(1,2);
R_sss = corrcoef(log10(N_values)', sss_means);
r_sss = R_sss(1,2);

% Simple t-test for correlation significance
n_pts = length(N_values);
t_th = r_th * sqrt(n_pts-2) / sqrt(1-r_th^2);
t_sss = r_sss * sqrt(n_pts-2) / sqrt(1-r_sss^2);

% Critical t-value for 95% (two-tailed, df=n-2)
t_crit = 2.571;  % for df=5, alpha=0.05

fprintf('Threshold: r=%.3f, |t|=%.2f ', r_th, abs(t_th));
if abs(t_th) < t_crit
    fprintf('(NO significant trend)\n');
    p_th = 0.1;  % placeholder
else
    fprintf('(SIGNIFICANT trend!)\n');
    p_th = 0.01;
end

fprintf('SSSgbKSE:  r=%.3f, |t|=%.2f ', r_sss, abs(t_sss));
if abs(t_sss) < t_crit
    fprintf('(NO significant trend)\n');
    p_sss = 0.1;
else
    fprintf('(SIGNIFICANT trend!)\n');
    p_sss = 0.01;
end

%% Plot with Error Bars
figure('Position', [100 100 1000 450], 'Color', 'w');

% Threshold
subplot(1,2,1);
th_mean = mean(ber_th_all, 2) * 100;
th_std = std(ber_th_all, 0, 2) * 100;
th_ci = 1.96 * th_std / sqrt(n_trials);  % 95% CI

errorbar(1:length(N_values), th_mean, th_ci, '-o', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r', 'Color', 'r', 'CapSize', 10);
hold on;
yline(mean(th_mean), 'r--', 'LineWidth', 1.5);
set(gca, 'XTick', 1:length(N_values), 'XTickLabel', N_values);
xlabel('Sequence Length N', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Threshold (|t|=%.2f)', abs(t_th)), 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

% SSSgbKSE
subplot(1,2,2);
sss_mean = mean(ber_sss_all, 2) * 100;
sss_std = std(ber_sss_all, 0, 2) * 100;
sss_ci = 1.96 * sss_std / sqrt(n_trials);

errorbar(1:length(N_values), sss_mean, sss_ci, '-s', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'g', 'Color', [0 0.6 0], 'CapSize', 10);
hold on;
yline(mean(sss_mean), '--', 'LineWidth', 1.5, 'Color', [0 0.6 0]);
set(gca, 'XTick', 1:length(N_values), 'XTickLabel', N_values);
xlabel('Sequence Length N', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('SSSgbKSE (|t|=%.2f)', abs(t_sss)), 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

sgtitle(sprintf('Effect of Sequence Length with 95%% CI (τ=%.1f, SNR=%d dB, %d trials)', ...
    tau, SNR_dB, n_trials), 'FontSize', 14, 'FontWeight', 'bold');

if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/n_effect_rigorous.png');
fprintf('\nSaved: figures/n_effect_rigorous.png\n');

%% Conclusion
fprintf('\n=== CONCLUSION ===\n');
if abs(t_th) < t_crit && abs(t_sss) < t_crit
    fprintf('✓ NO significant effect of sequence length N on BER\n');
    fprintf('✓ Simulations are reliable regardless of N choice\n');
    fprintf('✓ Advisor''s hypothesis NOT supported by data\n');
else
    fprintf('✗ Significant effect detected - investigate further\n');
end

%% Function
function out = SSSgbKSE(in, ISI, K)
    N = length(in); L = length(ISI);
    ssd = zeros(1,N); out = zeros(1,N);
    for n = 1:N
        if n == 1, ssd(n) = in(n);
        else
            c = 0;
            for k = 1:min(n-1,L-1)
                if k+1<=L, c = c + sign(ssd(n-k))*ISI(k+1); end
            end
            ssd(n) = in(n) - c;
        end
        if n > K
            for i = (n-K):n
                v = in(i);
                for k = 1:min(n-i,L-1)
                    if k+1<=L && i+k<=n, v = v - sign(ssd(i+k))*ISI(k+1); end
                end
                for k = 1:min(i-1,L-1)
                    if k+1<=L && i-k>=1, v = v - sign(out(i-k))*ISI(k+1); end
                end
                out(i) = v;
            end
        end
    end
    out(out==0) = ssd(out==0);
end