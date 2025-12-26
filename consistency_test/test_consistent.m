%% FTN Consistent Test - With Normalization

clear; clc; close all;

%% Parameters
tau = 0.7;
SNR_range = 0:2:14;
N = 10000;

sps = 10; span = 6;
h = rcosdesign(0.3, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;
half_win = 15;
num_fb = 4;
num_nb = 3;

% ISI
hh = conv(h, h);
[~, pk] = max(hh);
g = hh(pk:step:end); g = g/g(1); g = g(abs(g)>0.001);

fprintf('=== Consistent FTN Test (τ=%.1f, WITH normalization) ===\n\n', tau);

%% Load NN (normalizasyonla eğitilmiş model)
% Mevcut model dosyasını kullan
model_files = dir('mat/comparison/tau07*.mat');
if isempty(model_files)
    error('No model file found! Check mat/comparison/ folder');
end
load(fullfile('mat/comparison', model_files(1).name));
fprintf('Loaded: %s\n\n', model_files(1).name);

%% Results
n_snr = length(SNR_range);
ber_th = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);
ber_nb = zeros(1, n_snr);
ber_win = zeros(1, n_snr);

%% Main Loop - AYNI data, AYNI normalizasyon
for s = 1:n_snr
    SNR_dB = SNR_range(s);
    
    % Generate ONE dataset
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    
    tx = conv(upsample(symbols, step), h);
    
    % NOISE: Orijinal NN training ile aynı yöntem
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
    
    rx_mf = conv(rx_noisy, h);
    
    % NORMALIZATION: NN training ile aynı
    rx_mf_norm = rx_mf / std(rx_mf);
    
    % Sample
    rx_sym = zeros(1, N);
    rx_sym_norm = zeros(1, N);
    for i = 1:N
        idx = (i-1)*step + 1 + delay;
        if idx > 0 && idx <= length(rx_mf)
            rx_sym(i) = rx_mf(idx);           % unnormalized
            rx_sym_norm(i) = rx_mf_norm(idx); % normalized
        end
    end
    
    % === THRESHOLD (unnormalized - threshold = 0) ===
    err_th = sum((rx_sym > 0) ~= bits');
    ber_th(s) = err_th / N;
    
    % === SSSgbKSE (unnormalized - ISI coefficients based) ===
    det_sss = SSSgbKSE(rx_sym, g, 4) > 0;
    ber_sss(s) = sum(det_sss ~= bits') / N;
    
    % === NN: Neighbors+DF (NORMALIZED - NN ile tutarlı) ===
    err_nb = 0;
    count_nb = 0;
    df = zeros(num_fb, 1);
    for i = (num_nb+num_fb+1):N
        ctr = (i-1)*step + 1 + delay;
        if ctr - num_nb*step < 1 || ctr + num_nb*step > length(rx_mf_norm), continue; end
        
        % Features: center + left neighbors + right neighbors + df
        feat = rx_mf_norm(ctr);
        for k = 1:num_nb
            feat = [feat, rx_mf_norm(ctr - k*step)];
        end
        for k = 1:num_nb
            feat = [feat, rx_mf_norm(ctr + k*step)];
        end
        feat = [feat, df'];
        
        prob = predict(net4, feat);
        [~, pred] = max(prob);
        pred_bit = pred - 1;
        
        if pred_bit ~= bits(i), err_nb = err_nb + 1; end
        count_nb = count_nb + 1;
        df = [2*pred_bit-1; df(1:end-1)];
    end
    ber_nb(s) = err_nb / count_nb;
    
    % === NN: Window+DF (NORMALIZED) ===
    err_win = 0;
    count_win = 0;
    df = zeros(num_fb, 1);
    for i = (num_fb+1):N
        ctr = (i-1)*step + 1 + delay;
        if ctr - half_win < 1 || ctr + half_win > length(rx_mf_norm), continue; end
        
        % Features: window + df
        feat = rx_mf_norm(ctr-half_win : ctr+half_win)';
        feat = [feat, df'];
        
        prob = predict(net6, feat);
        [~, pred] = max(prob);
        pred_bit = pred - 1;
        
        if pred_bit ~= bits(i), err_win = err_win + 1; end
        count_win = count_win + 1;
        df = [2*pred_bit-1; df(1:end-1)];
    end
    ber_win(s) = err_win / count_win;
    
    fprintf('SNR=%2d: Th=%.2e, SSS=%.2e, Nb+DF=%.2e, Win+DF=%.2e\n', ...
        SNR_dB, ber_th(s), ber_sss(s), ber_nb(s), ber_win(s));
end

%% Results Summary
fprintf('\n============================================\n');
fprintf('=== SUMMARY @ 10 dB ===\n');
fprintf('============================================\n');
idx = find(SNR_range == 10);
fprintf('AWGN Theory:  %.2e\n', qfunc(sqrt(2*10^(10/10))));
fprintf('Threshold:    %.2e\n', ber_th(idx));
fprintf('SSSgbKSE:     %.2e\n', ber_sss(idx));
fprintf('Neighbors+DF: %.2e\n', ber_nb(idx));
fprintf('Window+DF:    %.2e\n', ber_win(idx));

fprintf('\n--- Comparisons ---\n');
if ber_sss(idx) > 0
    fprintf('SSSgbKSE vs Threshold: %.1fx better\n', ber_th(idx)/ber_sss(idx));
end
if ber_nb(idx) > 0 && ber_sss(idx) > 0
    fprintf('Neighbors+DF vs SSSgbKSE: %.1fx\n', ber_sss(idx)/ber_nb(idx));
end
if ber_win(idx) > 0 && ber_sss(idx) > 0
    fprintf('Window+DF vs SSSgbKSE: %.1fx\n', ber_sss(idx)/ber_win(idx));
end
if ber_win(idx) > 0 && ber_nb(idx) > 0
    fprintf('Window+DF vs Neighbors+DF: %.1fx\n', ber_nb(idx)/ber_win(idx));
end

%% Plot
figure('Position', [100 100 900 600]);
semilogy(SNR_range, qfunc(sqrt(2*10.^(SNR_range/10))), 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss, 'g-v', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_nb, 'm-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Neighbors+DF (NN)');
semilogy(SNR_range, ber_win, 'b-s', 'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Window+DF (NN)');
grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 10);
title(sprintf('FTN Detection Comparison (τ=%.1f)', tau), 'FontSize', 13);
ylim([1e-5 1]);

if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/consistent_comparison.png');

%% Save
save('mat/comparison/consistent_results.mat', 'SNR_range', 'ber_th', 'ber_sss', 'ber_nb', 'ber_win', 'tau');
fprintf('\nSaved results.\n');

%% SSSgbKSE Function
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