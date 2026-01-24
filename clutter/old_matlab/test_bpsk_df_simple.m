clear; clc; close all;

[file, path] = uigetfile('mat/bpsk_simple/*.mat');
if isequal(file, 0), error('No file selected'); end
load(fullfile(path, file));

fprintf('Testing BPSK DF Equalizer: tau=%.1f\n\n', tau);

SNR_range = 0:2:12;
ber_nn = zeros(size(SNR_range));
ber_threshold = zeros(size(SNR_range));
ber_theory_awgn = qfunc(sqrt(2 * 10.^(SNR_range/10)));

min_errors = 50;
max_bits = 1e5;

for idx = 1:length(SNR_range)
    snr = SNR_range(idx);
    if exist('num_feedback', 'var')
        [errors_nn, total, errors_threshold] = test_bpsk_df(net, tau, snr, window_len, num_feedback, min_errors, max_bits);
    else
        [errors_nn, total, errors_threshold] = test_bpsk(net, tau, snr, window_len, min_errors, max_bits);
    end
    ber_nn(idx) = errors_nn / total;
    ber_threshold(idx) = errors_threshold / total;
    fprintf('SNR=%2d dB: NN=%.4e, Threshold=%.4e (%d/%d errors)\n', ...
        snr, ber_nn(idx), ber_threshold(idx), errors_nn, total);
end

figure('Position', [100 100 800 500]);

semilogy(SNR_range, ber_theory_awgn, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory (AWGN)');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Threshold');
if exist('num_feedback', 'var')
    semilogy(SNR_range, ber_nn, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('DF-CNN (τ=%.1f)', tau));
else
    semilogy(SNR_range, ber_nn, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('NN (τ=%.1f)', tau));
end

grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
legend('Location', 'southwest');

if abs(tau - 1.0) < 0.01
    title('BPSK τ=1.0 (Nyquist - No ISI)');
else
    if exist('num_feedback', 'var')
        title(sprintf('BPSK DF Equalization (τ=%.1f, %d taps)', tau, num_feedback));
    else
        title(sprintf('BPSK Equalization (τ=%.1f with ISI)', tau));
    end
end

ylim([1e-6 0.5]);

if ~exist(fullfile(path, 'results'), 'dir'), mkdir(fullfile(path, 'results')); end
save(fullfile(path, 'results', strrep(file, '.mat', '_results.mat')), 'SNR_range', 'ber_nn', 'ber_threshold', 'ber_theory_awgn');

function [errors_nn, total_bits, errors_threshold] = test_bpsk_df(net, tau, SNR_dB, win_len, num_fb, min_err, max_bits)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);

    delay = span * sps + 1;
    half_win = floor(win_len/2);

    errors_nn = 0;
    errors_threshold = 0;
    total_bits = 0;

    decision_history = zeros(num_fb, 1);

    while errors_nn < min_err && total_bits < max_bits
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

        for i = (num_fb+1):N
            idx = (i-1)*round(sps*tau) + 1 + delay;

            if idx > half_win && idx + half_win <= length(rx)
                win = rx(idx-half_win:idx+half_win);
                features = [real(win(:))', decision_history'];

                probs = predict(net, features);
                [~, pred_class] = max(probs, [], 2);
                pred_bit_nn = pred_class - 1;

                if pred_bit_nn ~= bits(i)
                    errors_nn = errors_nn + 1;
                end

                center_sample = real(rx(idx));
                pred_bit_th = (center_sample > 0);
                if pred_bit_th ~= bits(i)
                    errors_threshold = errors_threshold + 1;
                end

                total_bits = total_bits + 1;

                decision_history = [pred_bit_nn; decision_history(1:end-1)];

                if errors_nn >= min_err || total_bits >= max_bits
                    return;
                end
            end
        end
    end
end

function [errors_nn, total_bits, errors_threshold] = test_bpsk(net, tau, SNR_dB, win_len, min_err, max_bits)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    delay = span * sps + 1;
    half_win = floor(win_len/2);
    
    errors_nn = 0;
    errors_threshold = 0;
    total_bits = 0;
    
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
    
    feature_buffer = [];
    bit_buffer = [];
    center_buffer = [];
    
    for i = 1:N
        idx = (i-1)*round(sps*tau) + 1 + delay;
        
        if idx > half_win && idx + half_win <= length(rx)
            win = rx(idx-half_win:idx+half_win);
            features = real(win(:))';
            center_sample = real(rx(idx));
            
            feature_buffer = [feature_buffer; features];
            bit_buffer = [bit_buffer; bits(i)];
            center_buffer = [center_buffer; center_sample];
            
            if size(feature_buffer, 1) >= 1000 || i == N
                probs = predict(net, feature_buffer);
                [~, pred_classes] = max(probs, [], 2);
                pred_bits_nn = pred_classes - 1;
                
                pred_bits_th = (center_buffer > 0);
                
                for j = 1:length(pred_bits_nn)
                    if pred_bits_nn(j) ~= bit_buffer(j)
                        errors_nn = errors_nn + 1;
                    end
                    
                    if pred_bits_th(j) ~= bit_buffer(j)
                        errors_threshold = errors_threshold + 1;
                    end
                    
                    total_bits = total_bits + 1;
                    
                    if errors_nn >= min_err
                        return;
                    end
                end
                
                feature_buffer = [];
                bit_buffer = [];
                center_buffer = [];
            end
        end
    end
end