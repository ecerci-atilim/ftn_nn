%% FTN Unified Comparison - NN vs Classical
% Tüm detektorlar AYNI koşullarda test edilir
% Power normalization tutarlı
% N etkisi analizi dahil
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== PARAMETERS ==========
tau = 0.7;
SNR_range = 0:2:10;
min_errors = 100;
max_bits = 1e6;

% CRITICAL: Test sequence length (N sweep için değiştir)
N_test = 1000;  % Test with moderate N first

% Power normalization - MUST MATCH TRAINING
USE_POWER_NORM = false;

%% ========== LOAD NN MODELS ==========
% Find correct file based on normalization setting
if USE_POWER_NORM
    model_file = sprintf('mat/comparison/tau%02d.mat', tau*10);
else
    model_file = sprintf('mat/comparison/tau%02d_nonorm.mat', tau*10);
end

if ~exist(model_file, 'file')
    error('Model file not found: %s\nRun train_comparison_corrected.m first!', model_file);
end

load(model_file);
fprintf('=== Unified FTN Comparison ===\n');
fprintf('τ = %.1f, N = %d, Power Norm = %s\n\n', tau, N_test, string(USE_POWER_NORM));

%% ========== SETUP ==========
sps = 10; span = 6;
h = rcosdesign(0.3, span, sps, 'sqrt');
h = h / norm(h);
step = round(sps * tau);
delay = span * sps;

% ISI for classical detectors
hh = conv(h, h);
[~, peak_idx] = max(hh);
g = hh(peak_idx : step : end);
g = g / g(1);
g = g(abs(g) > 0.001);

fprintf('ISI taps: %d\n', length(g));

%% ========== RESULTS STORAGE ==========
n_snr = length(SNR_range);
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

% NN models
ber_nn = zeros(7, n_snr);
names_nn = {'Single', 'Single+DF', 'Neighbors', 'Neighbors+DF', 'Window', 'Window+DF', 'Full'};

% Classical detectors
ber_threshold = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);

%% ========== TEST CLASSICAL DETECTORS ==========
fprintf('--- Testing Classical Detectors ---\n');

for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    EbN0_linear = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_linear);
    
    errors_th = 0;
    errors_sss = 0;
    total = 0;
    
    while (errors_th < min_errors || errors_sss < min_errors) && total < max_bits
        bits = randi([0 1], N_test, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx, h);
        
        % Apply same normalization as NN (if used)
        if USE_POWER_NORM
            rx_mf = rx_mf / std(rx_mf);
        end
        
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

%% ========== TEST NN MODELS ==========
fprintf('\n--- Testing NN Models ---\n');

models = {net1, net2, net3, net4, net5, net6, net7};
params = {
    [1, 0, 0, false];
    [1, num_feedback, 0, false];
    [1, 0, num_neighbor, true];
    [1, num_feedback, num_neighbor, true];
    [window_len, 0, 0, false];
    [window_len, num_feedback, 0, false];
    [window_len, num_feedback, num_neighbor, false];
};

% Only test selected models (Window+DF and Neighbors+DF for quick comparison)
test_models = [4, 6];  % Change to 1:7 for all models

for m = test_models
    fprintf('Model %d: %s\n', m, names_nn{m});
    p = params{m};
    
    for s_idx = 1:n_snr
        [err, total] = test_nn_model(models{m}, tau, SNR_range(s_idx), ...
            p(1), p(2), p(3), p(4), min_errors, max_bits, N_test, USE_POWER_NORM);
        ber_nn(m, s_idx) = err / total;
        fprintf('  %2d dB: %.2e\n', SNR_range(s_idx), ber_nn(m, s_idx));
    end
end

%% ========== PLOT RESULTS ==========
figure('Position', [100 100 900 600]);

% Theory
semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;

% Classical
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss, 'g-v', 'LineWidth', 1.5, 'DisplayName', 'SSSgbKSE');

% NN models
colors = {'c', 'c', 'm', 'm', 'b', 'b', 'k'};
markers = {'x', 'o', 'x', 'o', 's', 's', 'p'};
for m = test_models
    if ~any(isnan(ber_nn(m,:)))
        semilogy(SNR_range, ber_nn(m,:), [colors{m} '-' markers{m}], ...
            'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', names_nn{m});
    end
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 9);
title(sprintf('Unified Comparison (τ=%.1f, N=%d, Norm=%s)', tau, N_test, string(USE_POWER_NORM)));
ylim([1e-6 1]);

%% ========== SAVE ==========
if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, sprintf('figures/unified_tau%02d_N%d.png', tau*10, N_test));

if ~exist('mat/comparison', 'dir'), mkdir('mat/comparison'); end
save(sprintf('mat/comparison/unified_tau%02d_N%d.mat', tau*10, N_test), ...
    'SNR_range', 'ber_theory', 'ber_threshold', 'ber_sss', 'ber_nn', ...
    'names_nn', 'tau', 'N_test', 'USE_POWER_NORM');

%% ========== ANALYSIS ==========
fprintf('\n=== Analysis @ 10 dB ===\n');
idx10 = find(SNR_range == 10);
if ~isempty(idx10)
    fprintf('AWGN Theory: %.2e\n', ber_theory(idx10));
    fprintf('Threshold:   %.2e\n', ber_threshold(idx10));
    fprintf('SSSgbKSE:    %.2e\n', ber_sss(idx10));
    for m = test_models
        if ~isnan(ber_nn(m, idx10))
            fprintf('%s: %.2e\n', names_nn{m}, ber_nn(m, idx10));
        end
    end
    
    % Gains
    fprintf('\n--- Gains ---\n');
    fprintf('SSSgbKSE vs Threshold: %.1fx\n', ber_threshold(idx10)/ber_sss(idx10));
    for m = test_models
        if ~isnan(ber_nn(m, idx10))
            fprintf('%s vs SSSgbKSE: %.1fx\n', names_nn{m}, ber_sss(idx10)/ber_nn(m, idx10));
        end
    end
end

fprintf('\n=== Done ===\n');

%% ========== FUNCTIONS ==========

function [errors, total] = test_nn_model(net, tau, SNR_dB, win_len, num_fb, num_nb, neighbors_only, min_err, max_bits, N, use_norm)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    EbN0_linear = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_linear);
    
    errors = 0;
    total = 0;
    
    while errors < min_err && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx = conv(rx, h);
        
        % CRITICAL: Same normalization as training
        if use_norm
            rx = rx / std(rx);
        end
        
        df_history = zeros(num_fb, 1);
        start_idx = max([num_fb + 1, num_nb + 1, 1]);
        
        if num_fb > 0
            % Sequential (DFE models)
            for i = start_idx:N
                center = (i - 1) * step + 1 + delay;
                if center - half_win < 1 || center + half_win > length(rx), continue; end
                if num_nb > 0 && (center - num_nb*step < 1 || center + num_nb*step > length(rx)), continue; end
                
                feat = build_features(rx, center, half_win, win_len, step, num_nb, df_history, neighbors_only);
                
                probs = predict(net, feat);
                [~, pred_class] = max(probs);
                pred_bit = pred_class - 1;
                pred_sym = 2 * pred_bit - 1;
                
                if pred_bit ~= bits(i), errors = errors + 1; end
                total = total + 1;
                
                df_history = [pred_sym; df_history(1:end-1)];
                
                if errors >= min_err || total >= max_bits, return; end
            end
        else
            % Batch (non-DFE models)
            feat_buf = [];
            bit_buf = [];
            
            for i = start_idx:N
                center = (i - 1) * step + 1 + delay;
                if center - half_win < 1 || center + half_win > length(rx), continue; end
                if num_nb > 0 && (center - num_nb*step < 1 || center + num_nb*step > length(rx)), continue; end
                
                feat = build_features(rx, center, half_win, win_len, step, num_nb, [], neighbors_only);
                feat_buf = [feat_buf; feat];
                bit_buf = [bit_buf; bits(i)];
                
                if size(feat_buf, 1) >= 1000 || i == N
                    probs = predict(net, feat_buf);
                    [~, pred_classes] = max(probs, [], 2);
                    pred_bits = pred_classes - 1;
                    
                    errors = errors + sum(pred_bits ~= bit_buf);
                    total = total + length(bit_buf);
                    
                    if errors >= min_err || total >= max_bits, return; end
                    
                    feat_buf = [];
                    bit_buf = [];
                end
            end
        end
    end
end

function feat = build_features(rx, center, half_win, win_len, step, num_nb, df_history, neighbors_only)
    if neighbors_only
        win_feat = zeros(1, 1 + 2*num_nb);
        win_feat(1) = rx(center);
        for k = 1:num_nb
            win_feat(1 + k) = rx(center - k*step);
            win_feat(1 + num_nb + k) = rx(center + k*step);
        end
        nb_feat = [];
    else
        if win_len == 1
            win_feat = rx(center);
        else
            win_feat = rx(center - half_win : center + half_win)';
        end
        
        if num_nb > 0
            nb_feat = zeros(1, 2 * num_nb);
            for k = 1:num_nb
                nb_feat(k) = rx(center - k * step);
                nb_feat(num_nb + k) = rx(center + k * step);
            end
        else
            nb_feat = [];
        end
    end
    
    if ~isempty(df_history)
        df_feat = df_history';
    else
        df_feat = [];
    end
    
    feat = [win_feat, nb_feat, df_feat];
end

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
