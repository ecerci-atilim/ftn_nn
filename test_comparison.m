clear; clc; close all;

[file, path] = uigetfile('mat/comparison/*.mat');
if isequal(file, 0), error('No file'); end
load(fullfile(path, file));

fprintf('=== Testing τ=%.1f ===\n\n', tau);

SNR_range = 0:2:14;
n_snr = length(SNR_range);

ber = zeros(5, n_snr);  % 5 model
ber_th = zeros(1, n_snr);
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

min_err = 100;
max_bits = 1e6;

models = {net1, net2, net3, net4, net5};
win_lens = [window_len, window_len, 1, 1, window_len];
num_fbs = [0, num_feedback, num_feedback, 0, num_feedback];
num_nbs = [0, 0, 0, 0, num_neighbor];
names = {'Window', 'Window+DF', 'Single+DF', 'Single', 'Full'};

for m = 1:5
    fprintf('Model %d: %s\n', m, names{m});
    for s = 1:n_snr
        [err, total] = test_model(models{m}, tau, SNR_range(s), ...
            win_lens(m), num_fbs(m), num_nbs(m), min_err, max_bits);
        ber(m, s) = err / total;
        fprintf('  %2d dB: %.2e\n', SNR_range(s), ber(m, s));
    end
end

fprintf('\nThreshold\n');
for s = 1:n_snr
    [err, total] = test_threshold(tau, SNR_range(s), min_err, max_bits);
    ber_th(s) = err / total;
    fprintf('  %2d dB: %.2e\n', SNR_range(s), ber_th(s));
end

%% Plot
figure('Position', [100 100 900 550]);
semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber(4,:), 'c-x', 'LineWidth', 1.5, 'DisplayName', 'Single');
semilogy(SNR_range, ber(3,:), 'm-d', 'LineWidth', 1.5, 'DisplayName', 'Single+DF');
semilogy(SNR_range, ber(1,:), 'b-o', 'LineWidth', 2, 'DisplayName', 'Window');
semilogy(SNR_range, ber(2,:), 'g-s', 'LineWidth', 2, 'DisplayName', 'Window+DF');
semilogy(SNR_range, ber(5,:), 'k-p', 'LineWidth', 2.5, 'MarkerSize', 9, 'DisplayName', 'Full');
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
legend('Location', 'southwest');
title(sprintf('FTN Detection Comparison (τ=%.1f)', tau));
ylim([1e-6 0.5]);

%% Analysis
idx10 = find(SNR_range == 10);
if ~isempty(idx10)
    fprintf('\n=== 10 dB Analysis ===\n');
    base = ber(4, idx10);
    for m = 1:5
        gain = base / ber(m, idx10);
        fprintf('%s: %.2e (%.1fx)\n', names{m}, ber(m, idx10), gain);
    end
    fprintf('\nWindow value: %.1fx\n', ber(4,idx10)/ber(1,idx10));
    fprintf('DF value: %.1fx\n', ber(4,idx10)/ber(3,idx10));
    fprintf('Window+DF vs Single+DF: %.1fx\n', ber(3,idx10)/ber(2,idx10));
end

%% Save
save(fullfile(path, strrep(file, '.mat', '_results.mat')), ...
    'SNR_range', 'ber', 'ber_th', 'ber_theory', 'names', 'tau');

%% Functions
function [errors, total] = test_model(net, tau, SNR_dB, win_len, num_fb, num_nb, min_err, max_bits)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    errors = 0;
    total = 0;
    
    while errors < min_err && total < max_bits
        N = 50000;
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        noise_var = 1 / (2 * 10^(SNR_dB/10));
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx = conv(rx, h);
        rx = rx / std(rx);
        
        df_history = zeros(num_fb, 1);
        start_idx = max([num_fb + 1, num_nb + 1, 1]);
        
        if num_fb > 0
            % Sequential (DF needs previous decisions)
            for i = start_idx:N
                center = (i - 1) * step + 1 + delay;
                if center - half_win < 1 || center + half_win > length(rx), continue; end
                
                feat = build_features(rx, center, half_win, win_len, step, num_nb, df_history);
                
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
            % Batch (no DF)
            feat_buf = [];
            bit_buf = [];
            
            for i = start_idx:N
                center = (i - 1) * step + 1 + delay;
                if center - half_win < 1 || center + half_win > length(rx), continue; end
                
                feat = build_features(rx, center, half_win, win_len, step, num_nb, []);
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

function feat = build_features(rx, center, half_win, win_len, step, num_nb, df_history)
    % Window
    if win_len == 1
        win_feat = rx(center);
    else
        win_feat = rx(center - half_win : center + half_win)';
    end
    
    % Neighbors
    if num_nb > 0
        nb_feat = zeros(1, 2 * num_nb);
        for k = 1:num_nb
            left_idx = center - k * step;
            right_idx = center + k * step;
            if left_idx > 0, nb_feat(k) = rx(left_idx); end
            if right_idx <= length(rx), nb_feat(num_nb + k) = rx(right_idx); end
        end
    else
        nb_feat = [];
    end
    
    % DF
    if ~isempty(df_history)
        df_feat = df_history';
    else
        df_feat = [];
    end
    
    feat = [win_feat, nb_feat, df_feat];
end

function [errors, total] = test_threshold(tau, SNR_dB, min_err, max_bits)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    step = round(sps * tau);
    delay = span * sps;
    
    errors = 0;
    total = 0;
    
    while errors < min_err && total < max_bits
        N = 50000;
        bits = randi([0 1], N, 1);
        symbols = 2 * bits - 1;
        
        tx = conv(upsample(symbols, step), h);
        noise_var = 1 / (2 * 10^(SNR_dB/10));
        rx = tx + sqrt(noise_var) * randn(size(tx));
        rx = conv(rx, h);
        
        for i = 1:N
            idx = (i - 1) * step + 1 + delay;
            if idx < 1 || idx > length(rx), continue; end
            
            pred_bit = real(rx(idx)) > 0;
            if pred_bit ~= bits(i), errors = errors + 1; end
            total = total + 1;
            
            if errors >= min_err || total >= max_bits, return; end
        end
    end
end