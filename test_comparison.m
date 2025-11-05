clear; clc; close all;

[file, path] = uigetfile('mat/comparison/*.mat');
if isequal(file, 0), error('No file selected'); end
load(fullfile(path, file));

fprintf('=== Comparing Four Approaches for τ=%.1f ===\n\n', tau);

SNR_range = 0:2:12;
ber1 = zeros(size(SNR_range));
ber2 = zeros(size(SNR_range));
ber3 = zeros(size(SNR_range));
ber4 = zeros(size(SNR_range));
ber_threshold = zeros(size(SNR_range));
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

min_errors = 100;
max_bits = 1e6;

fprintf('Model 1: Window ONLY (31 samples, no DF)\n');
for idx = 1:length(SNR_range)
    [err, total] = test_model(net1, tau, SNR_range(idx), window_len, 0, min_errors, max_bits);
    ber1(idx) = err / total;
    fprintf('  SNR=%2d dB: BER=%.4e\n', SNR_range(idx), ber1(idx));
end

fprintf('\nModel 2: Window + DF (31 samples + 4 DF taps)\n');
for idx = 1:length(SNR_range)
    [err, total] = test_model(net2, tau, SNR_range(idx), window_len, num_feedback, min_errors, max_bits);
    ber2(idx) = err / total;
    fprintf('  SNR=%2d dB: BER=%.4e\n', SNR_range(idx), ber2(idx));
end

fprintf('\nModel 3: Single Sample + DF (1 sample + 4 DF taps)\n');
for idx = 1:length(SNR_range)
    [err, total] = test_model(net3, tau, SNR_range(idx), 1, num_feedback, min_errors, max_bits);
    ber3(idx) = err / total;
    fprintf('  SNR=%2d dB: BER=%.4e\n', SNR_range(idx), ber3(idx));
end

fprintf('\nModel 4: Single Sample ONLY (1 sample, no DF)\n');
for idx = 1:length(SNR_range)
    [err, total] = test_model(net4, tau, SNR_range(idx), 1, 0, min_errors, max_bits);
    ber4(idx) = err / total;
    fprintf('  SNR=%2d dB: BER=%.4e\n', SNR_range(idx), ber4(idx));
end

fprintf('\nThreshold (uncoded FTN baseline)\n');
for idx = 1:length(SNR_range)
    [~, total, err_th] = test_threshold(tau, SNR_range(idx), min_errors, max_bits);
    ber_threshold(idx) = err_th / total;
    fprintf('  SNR=%2d dB: BER=%.4e\n', SNR_range(idx), ber_threshold(idx));
end

figure('Position', [100 100 1000 600]);
semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'Theory (AWGN)');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Threshold (uncoded FTN)');
semilogy(SNR_range, ber4, 'c-x', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Single ONLY (1 input)');
semilogy(SNR_range, ber3, 'm-d', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Single + DF (5 inputs)');
semilogy(SNR_range, ber1, 'b-o', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Window ONLY (31 inputs)');
semilogy(SNR_range, ber2, 'g-s', 'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'Window + DF (35 inputs) - BEST');
grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 10);
title(sprintf('Comparison: Window vs DF Analysis (τ=%.1f)', tau), 'FontSize', 13);
ylim([1e-6 0.5]);

fprintf('\n=== Gain Analysis at 10 dB ===\n');
fprintf('Single ONLY:     %.4e  (baseline - worst)\n', ber4(SNR_range==10));
fprintf('Single + DF:     %.4e  (%.1fx better - DF helps!)\n', ber3(SNR_range==10), ber4(SNR_range==10)/ber3(SNR_range==10));
fprintf('Window ONLY:     %.4e  (%.1fx better - window helps!)\n', ber1(SNR_range==10), ber4(SNR_range==10)/ber1(SNR_range==10));
fprintf('Window + DF:     %.4e  (%.1fx better - BEST!)\n', ber2(SNR_range==10), ber4(SNR_range==10)/ber2(SNR_range==10));
fprintf('\n=== Key Comparisons ===\n');
fprintf('Window value: Window ONLY vs Single ONLY = %.1fx gain\n', ber4(SNR_range==10)/ber1(SNR_range==10));
fprintf('DF value:     Single+DF vs Single ONLY = %.1fx gain\n', ber4(SNR_range==10)/ber3(SNR_range==10));
fprintf('Combined:     Window+DF vs Single+DF = %.1fx gain (proves window matters!)\n', ber3(SNR_range==10)/ber2(SNR_range==10));
fprintf('\nConclusion: Window+DF provides %.1fx gain over professor''s Single+DF approach\n', ber3(SNR_range==10)/ber2(SNR_range==10));

if ~exist(fullfile(path, 'results'), 'dir'), mkdir(fullfile(path, 'results')); end
save(fullfile(path, 'results', strrep(file, '.mat', '_results.mat')), ...
    'SNR_range', 'ber1', 'ber2', 'ber3', 'ber4', 'ber_threshold', 'ber_theory');

function [errors, total_bits] = test_model(net, tau, SNR_dB, win_len, num_fb, min_err, max_bits)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    delay = span * sps + 1;
    half_win = floor(win_len/2);
    
    errors = 0;
    total_bits = 0;
    decision_history = zeros(num_fb, 1);
    
    while errors < min_err && total_bits < max_bits
        N = 50000;
        bits = randi([0 1], N, 1);
        symbols = 2*bits - 1;
        
        tx = upsample(symbols, round(sps*tau));
        tx = conv(tx, h);
        
        EbN0_linear = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_linear);
        noise = sqrt(noise_var) * randn(size(tx));
        rx = tx + noise;
        rx = conv(rx, h);
        
        start_idx = max(num_fb+1, 1);
        
        if num_fb > 0
            for i = start_idx:N
                idx = (i-1)*round(sps*tau) + 1 + delay;
                
                if idx > half_win && idx + half_win <= length(rx)
                    if win_len == 1
                        win_features = real(rx(idx));
                    else
                        win = rx(idx-half_win:idx+half_win);
                        win_features = real(win(:))';
                    end
                    
                    features = [win_features, decision_history'];
                    
                    probs = predict(net, features);
                    [~, pred_class] = max(probs);
                    pred_bit = pred_class - 1;
                    
                    if pred_bit ~= bits(i)
                        errors = errors + 1;
                    end
                    total_bits = total_bits + 1;
                    
                    decision_history = [pred_bit; decision_history(1:end-1)];
                    
                    if errors >= min_err || total_bits >= max_bits
                        return;
                    end
                end
            end
        else
            feature_buffer = [];
            bit_buffer = [];
            
            for i = start_idx:N
                idx = (i-1)*round(sps*tau) + 1 + delay;
                
                if idx > half_win && idx + half_win <= length(rx)
                    if win_len == 1
                        win_features = real(rx(idx));
                    else
                        win = rx(idx-half_win:idx+half_win);
                        win_features = real(win(:))';
                    end
                    
                    feature_buffer = [feature_buffer; win_features];
                    bit_buffer = [bit_buffer; bits(i)];
                    
                    if size(feature_buffer, 1) >= 1000 || i == N
                        probs = predict(net, feature_buffer);
                        [~, pred_classes] = max(probs, [], 2);
                        pred_bits = pred_classes - 1;
                        
                        for j = 1:length(pred_bits)
                            if pred_bits(j) ~= bit_buffer(j)
                                errors = errors + 1;
                            end
                            total_bits = total_bits + 1;
                            
                            if errors >= min_err || total_bits >= max_bits
                                return;
                            end
                        end
                        
                        feature_buffer = [];
                        bit_buffer = [];
                    end
                end
            end
        end
    end
end

function [total_bits, total_bits_out, errors] = test_threshold(tau, SNR_dB, min_err, max_bits)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    delay = span * sps + 1;
    errors = 0;
    total_bits_out = 0;
    
    N = 50000;
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    
    tx = upsample(symbols, round(sps*tau));
    tx = conv(tx, h);
    
    EbN0_linear = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_linear);
    noise = sqrt(noise_var) * randn(size(tx));
    rx = tx + noise;
    rx = conv(rx, h);
    
    for i = 1:N
        idx = (i-1)*round(sps*tau) + 1 + delay;
        if idx > 0 && idx <= length(rx)
            pred_bit = (real(rx(idx)) > 0);
            if pred_bit ~= bits(i)
                errors = errors + 1;
            end
            total_bits_out = total_bits_out + 1;
            if errors >= min_err
                break;
            end
        end
    end
    total_bits = total_bits_out;
end