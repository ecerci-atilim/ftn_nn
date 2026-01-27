%% FTN Full Comparison - Corrected SNR

clear; clc; close all;

%% ========== PARAMETERS ==========
tau = 0.7;
SNR_range = 0:2:14;
N_test = 1000;
min_errors = 100;
max_bits = 1e6;

SNR_METHOD = 'corrected';

%% ========== LOAD NN MODELS ==========
model_file = sprintf('mat/comparison/tau%02d_snr_corrected.mat', tau*10);
if ~exist(model_file, 'file')
    error('Model file not found: %s\nRun train_snr_corrected.m first!', model_file);
end
load(model_file);

fprintf('=== Full FTN Comparison (Corrected SNR) ===\n');
fprintf('τ = %.1f, N = %d\n\n', tau, N_test);

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

fprintf('ISI taps: %d\n\n', length(g));

%% ========== RESULTS ==========
n_snr = length(SNR_range);
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

% Classical
ber_threshold = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);

% NN (only test key models)
ber_neighbors_df = zeros(1, n_snr);  % Model 4
ber_window_df = zeros(1, n_snr);     % Model 6

%% ========== TEST CLASSICAL ==========
fprintf('--- Classical Detectors ---\n');

for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    EbN0_linear = 10^(SNR_dB/10);
    
    errors_th = 0;
    errors_sss = 0;
    total = 0;
    
    while (errors_th < min_errors || errors_sss < min_errors) && total < max_bits
        bits = randi([0 1], N_test, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        
        % CORRECTED SNR
        sig_pwr = mean(tx.^2);
        noise_var = sig_pwr / (2 * EbN0_linear);
        
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx_mf = conv(rx, h);
        
        % Sample
        rx_sampled = zeros(1, N_test);
        for i = 1:N_test
            idx = (i-1) * step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf)
                rx_sampled(i) = real(rx_mf(idx));
            end
        end
        
        % Threshold
        bits_th = rx_sampled > 0;
        errors_th = errors_th + sum(bits_th ~= bits');
        
        % SSSgbKSE
        bits_sss = SSSgbKSE(rx_sampled, g, 4) > 0;
        errors_sss = errors_sss + sum(bits_sss ~= bits');
        
        total = total + N_test;
    end
    
    ber_threshold(s_idx) = errors_th / total;
    ber_sss(s_idx) = errors_sss / total;
    
    fprintf('SNR=%2d: Thresh=%.2e, SSS=%.2e\n', SNR_dB, ber_threshold(s_idx), ber_sss(s_idx));
end

%% ========== TEST NN: Neighbors+DF (Model 4) ==========
fprintf('\n--- NN: Neighbors+DF ---\n');

for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    [err, total] = test_nn(net4, tau, SNR_dB, 1, num_feedback, num_neighbor, true, min_errors, max_bits, N_test, SNR_METHOD);
    ber_neighbors_df(s_idx) = err / total;
    fprintf('SNR=%2d: %.2e\n', SNR_dB, ber_neighbors_df(s_idx));
end

%% ========== TEST NN: Window+DF (Model 6) ==========
fprintf('\n--- NN: Window+DF ---\n');

for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    [err, total] = test_nn(net6, tau, SNR_dB, window_len, num_feedback, 0, false, min_errors, max_bits, N_test, SNR_METHOD);
    ber_window_df(s_idx) = err / total;
    fprintf('SNR=%2d: %.2e\n', SNR_dB, ber_window_df(s_idx));
end

%% ========== PLOT ==========
figure('Position', [100 100 900 600]);

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Uncoded FTN');
semilogy(SNR_range, ber_sss, 'g-v', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_neighbors_df, 'm-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Neighbors+DF (NN)');
semilogy(SNR_range, ber_window_df, 'b-s', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Window+DF (NN)');

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 10);
title(sprintf('FTN Detection Comparison - Corrected SNR (τ=%.1f)', tau), 'FontSize', 13);
ylim([1e-6 1]);

if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/full_comparison_corrected.png');

%% ========== ANALYSIS ==========
fprintf('\n========================================\n');
fprintf('=== ANALYSIS @ 10 dB ===\n');
fprintf('========================================\n');

idx10 = find(SNR_range == 10);
if ~isempty(idx10)
    fprintf('AWGN Theory:     %.2e\n', ber_theory(idx10));
    fprintf('Uncoded FTN:     %.2e\n', ber_threshold(idx10));
    fprintf('SSSgbKSE:        %.2e\n', ber_sss(idx10));
    fprintf('Neighbors+DF:    %.2e\n', ber_neighbors_df(idx10));
    fprintf('Window+DF:       %.2e\n', ber_window_df(idx10));
    
    fprintf('\n--- Gains ---\n');
    fprintf('SSSgbKSE vs Uncoded:     %.1fx\n', ber_threshold(idx10)/ber_sss(idx10));
    fprintf('Neighbors+DF vs SSSgbKSE: %.1fx\n', ber_sss(idx10)/ber_neighbors_df(idx10));
    fprintf('Window+DF vs SSSgbKSE:    %.1fx\n', ber_sss(idx10)/ber_window_df(idx10));
    fprintf('Window+DF vs Neighbors+DF: %.1fx\n', ber_neighbors_df(idx10)/ber_window_df(idx10));
end

%% ========== SAVE ==========
save('mat/comparison/full_corrected.mat', ...
    'SNR_range', 'ber_theory', 'ber_threshold', 'ber_sss', ...
    'ber_neighbors_df', 'ber_window_df', 'tau', 'N_test');

fprintf('\nSaved: mat/comparison/full_corrected.mat\n');

%% ========== FUNCTIONS ==========

function [errors, total] = test_nn(net, tau, SNR_dB, win_len, num_fb, num_nb, neighbors_only, min_err, max_bits, N, snr_method)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    EbN0_linear = 10^(SNR_dB/10);
    
    errors = 0;
    total = 0;
    
    while errors < min_err && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        
        % CORRECTED SNR
        if strcmp(snr_method, 'corrected')
            sig_pwr = mean(tx.^2);
            noise_var = sig_pwr / (2 * EbN0_linear);
        else
            noise_var = 1 / (2 * EbN0_linear);
        end
        
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx = conv(rx, h);
        
        df_history = zeros(num_fb, 1);
        start_idx = max([num_fb + 1, num_nb + 1, 1]);
        
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
            
            if num_fb > 0
                df_history = [pred_sym; df_history(1:end-1)];
            end
            
            if errors >= min_err || total >= max_bits, return; end
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
