%% FTN Quick Test - Faster Version
% Azaltılmış parametreler ile hızlı test
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== PARAMETERS (REDUCED) ==========
tau = 0.7;
SNR_range = 0:2:14;
N_test = 500;       % REDUCED
min_errors = 50;    % REDUCED
max_bits = 2e5;     % REDUCED

%% ========== LOAD ==========
model_file = sprintf('mat/comparison/tau%02d_awgn.mat', tau*10);
if ~exist(model_file, 'file')
    error('Run train_awgn.m first!');
end
load(model_file);

fprintf('=== Quick FTN Test ===\n');
fprintf('τ=%.1f, N=%d, min_err=%d\n\n', tau, N_test, min_errors);

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

%% ========== RESULTS ==========
n_snr = length(SNR_range);
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));
ber_threshold = zeros(1, n_snr);
ber_sss = zeros(1, n_snr);
ber_neighbors_df = zeros(1, n_snr);
ber_window_df = zeros(1, n_snr);

%% ========== CLASSICAL (fast) ==========
fprintf('Classical... ');
for s_idx = 1:n_snr
    SNR_dB = SNR_range(s_idx);
    err_th = 0; err_sss = 0; total = 0;
    
    while (err_th < min_errors || err_sss < min_errors) && total < max_bits
        bits = randi([0 1], N_test, 1);
        symbols = 2 * bits - 1;
        tx = conv(upsample(symbols, step), h);
        rx = awgn(tx, SNR_dB, 'measured');
        rx_mf = conv(rx, h);
        
        rx_s = zeros(1, N_test);
        for i = 1:N_test
            idx = (i-1)*step + 1 + delay;
            if idx > 0 && idx <= length(rx_mf), rx_s(i) = real(rx_mf(idx)); end
        end
        
        err_th = err_th + sum((rx_s > 0) ~= bits');
        err_sss = err_sss + sum((SSSgbKSE(rx_s, g, 4) > 0) ~= bits');
        total = total + N_test;
    end
    ber_threshold(s_idx) = err_th / total;
    ber_sss(s_idx) = err_sss / total;
end
fprintf('Done\n');

%% ========== NN: Neighbors+DF ==========
fprintf('Neighbors+DF... ');
for s_idx = 1:n_snr
    [err, total] = test_nn_fast(net4, tau, SNR_range(s_idx), 1, num_feedback, num_neighbor, true, min_errors, max_bits, N_test);
    ber_neighbors_df(s_idx) = err / total;
end
fprintf('Done\n');

%% ========== NN: Window+DF ==========
fprintf('Window+DF... ');
for s_idx = 1:n_snr
    [err, total] = test_nn_fast(net6, tau, SNR_range(s_idx), window_len, num_feedback, 0, false, min_errors, max_bits, N_test);
    ber_window_df(s_idx) = err / total;
end
fprintf('Done\n');

%% ========== RESULTS TABLE ==========
fprintf('\n=== BER Results ===\n');
fprintf('SNR\tTheory\t\tThresh\t\tSSS\t\tNb+DF\t\tWin+DF\n');
for s = 1:n_snr
    fprintf('%d\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n', ...
        SNR_range(s), ber_theory(s), ber_threshold(s), ber_sss(s), ...
        ber_neighbors_df(s), ber_window_df(s));
end

%% ========== KEY COMPARISON @ 10 dB ==========
idx10 = find(SNR_range == 10);
fprintf('\n=== @ 10 dB ===\n');
fprintf('SSSgbKSE:     %.2e\n', ber_sss(idx10));
fprintf('Neighbors+DF: %.2e\n', ber_neighbors_df(idx10));
fprintf('Window+DF:    %.2e\n', ber_window_df(idx10));

if ber_window_df(idx10) > 0 && ber_neighbors_df(idx10) > 0
    fprintf('\n*** Window+DF vs Neighbors+DF: %.1fx ***\n', ...
        ber_neighbors_df(idx10) / ber_window_df(idx10));
end

%% ========== PLOT ==========
figure('Position', [100 100 900 600]);
semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Uncoded');
semilogy(SNR_range, ber_sss, 'g-v', 'LineWidth', 1.5, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_neighbors_df, 'm-o', 'LineWidth', 2, 'DisplayName', 'Neighbors+DF');
semilogy(SNR_range, ber_window_df, 'b-s', 'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Window+DF');
grid on; xlabel('SNR (dB)'); ylabel('BER');
legend('Location', 'southwest');
title(sprintf('FTN Detection (τ=%.1f)', tau));
ylim([1e-5 1]);

%% ========== FUNCTIONS ==========
function [errors, total] = test_nn_fast(net, tau, SNR_dB, win_len, num_fb, num_nb, neighbors_only, min_err, max_bits, N)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    errors = 0; total = 0;
    
    while errors < min_err && total < max_bits
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        tx = conv(upsample(symbols, step), h);
        rx = awgn(tx, SNR_dB, 'measured');
        rx = conv(rx, h);
        
        df_hist = zeros(num_fb, 1);
        start_idx = max([num_fb + 1, num_nb + 1, 1]);
        
        for i = start_idx:N
            center = (i-1)*step + 1 + delay;
            if center - half_win < 1 || center + half_win > length(rx), continue; end
            if num_nb > 0 && (center - num_nb*step < 1 || center + num_nb*step > length(rx)), continue; end
            
            % Build features
            if neighbors_only
                feat = [rx(center), rx(center - (1:num_nb)*step), rx(center + (1:num_nb)*step)];
            else
                feat = rx(center - half_win : center + half_win)';
            end
            if num_fb > 0, feat = [feat, df_hist']; end
            
            % Predict
            probs = predict(net, feat);
            [~, pred_class] = max(probs);
            pred_bit = pred_class - 1;
            
            if pred_bit ~= bits(i), errors = errors + 1; end
            total = total + 1;
            
            if num_fb > 0
                df_hist = [2*pred_bit-1; df_hist(1:end-1)];
            end
            
            if errors >= min_err || total >= max_bits, return; end
        end
    end
end

function mesSE = SSSgbKSE(mesin, ISI, gbK)
    s = length(mesin); L = length(ISI);
    ssd = zeros(1,s); mesSE = zeros(1,s);
    for n = 1:s
        if n == 1, ssd(n) = mesin(n);
        else
            isi_c = 0;
            for k = 1:min(n-1, L-1)
                if k+1 <= L, isi_c = isi_c + sign(ssd(n-k))*ISI(k+1); end
            end
            ssd(n) = mesin(n) - isi_c;
        end
        if n > gbK
            for i = (n-gbK):n
                val = mesin(i);
                for k = 1:min(n-i, L-1)
                    if k+1 <= L && i+k <= n, val = val - sign(ssd(i+k))*ISI(k+1); end
                end
                for k = 1:min(i-1, L-1)
                    if k+1 <= L && i-k >= 1, val = val - sign(mesSE(i-k))*ISI(k+1); end
                end
                mesSE(i) = val;
            end
        end
    end
    mesSE(mesSE == 0) = ssd(mesSE == 0);
end