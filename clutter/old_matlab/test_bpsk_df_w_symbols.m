clear; clc; close all;

[file, path] = uigetfile('mat/bpsk_simple/*.mat');
if isequal(file, 0), error('No file selected'); end
% This will load net, tau, window_len, num_feedback, num_past_future_symbols, input_size
load(fullfile(path, file)); 

fprintf('Testing BPSK DF Equalizer: tau=%.1f\n\n', tau);
fprintf('Model trained with %d feedback taps and %d past/future symbols as features.\n', num_feedback, num_past_future_symbols);

SNR_range = 0:2:12;
ber_nn = zeros(size(SNR_range));
ber_threshold = zeros(size(SNR_range));
ber_theory_awgn = qfunc(sqrt(2 * 10.^(SNR_range/10)));

min_errors = 50;
max_bits = 1e5;

for idx = 1:length(SNR_range)
    snr = SNR_range(idx);
    % Pass num_past_future_symbols to the test_bpsk_df function
    % Assuming a DF model is always loaded for this script
    [errors_nn, total, errors_threshold] = test_bpsk_df(net, tau, snr, window_len, num_feedback, num_past_future_symbols, min_errors, max_bits);

    ber_nn(idx) = errors_nn / total;
    ber_threshold(idx) = errors_threshold / total;
    fprintf('SNR=%2d dB: NN=%.4e, Threshold=%.4e (%d/%d errors)\n', ...
        snr, ber_nn(idx), ber_threshold(idx), errors_nn, total);
end

figure('Position', [100 100 800 500]);

semilogy(SNR_range, ber_theory_awgn, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory (AWGN)');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Uncoded FTN');
% Updated legend for DF with extended features
semilogy(SNR_range, ber_nn, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('DF-CNN (τ=%.1f, %d taps)', tau, num_feedback));

grid on;
xlabel('$E_b$/$N_0$ (dB)');
ylabel('BER');
legend('Location', 'southwest');

% if abs(tau - 1.0) < 0.01
%     title('BPSK DF Equalization τ=1.0 (Nyquist - No ISI) with Extended Features');
% else
%     title(sprintf('BPSK DF Equalization (τ=%.1f, %d taps) with Extended Features', tau, num_feedback));
% end

ylim([1e-6 0.5]);

if ~exist(fullfile(path, 'results'), 'dir'), mkdir(fullfile(path, 'results')); end
save(fullfile(path, 'results', strrep(file, '.mat', '_results.mat')), 'SNR_range', 'ber_nn', 'ber_threshold', 'ber_theory_awgn');


% This function has been updated to include num_past_future_symbols in its features
function [errors_nn, total_bits, errors_threshold] = test_bpsk_df(net, tau, SNR_dB, win_len, num_fb, num_past_future_symbols, min_err, max_bits)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);

    delay_system = span * sps + 1; % System delay from filters
    symbol_spacing_samples = round(sps*tau); % Delay between symbols in samples

    half_win = floor(win_len/2);

    errors_nn = 0;
    errors_threshold = 0;
    total_bits = 0;

    % Initialize decision history (assuming all zeros before the first symbol)
    decision_history = zeros(num_fb, 1);

    % Loop until enough errors or total bits are accumulated
    while errors_nn < min_err && total_bits < max_bits
        N = 50000; % Number of symbols to generate in this iteration
        bits = randi([0 1], N, 1);
        symbols = 2*bits - 1;

        tx = upsample(symbols, symbol_spacing_samples);
        tx = conv(tx, h);

        EbN0_linear = 10^(SNR_dB/10);
        noise_var = 1 / (2 * EbN0_linear);
        noise = sqrt(noise_var) * randn(size(tx));
        rx = tx + noise;

        rx = conv(rx, h);

        % Loop through received symbols to make decisions
        for i = (num_fb + num_past_future_symbols + 1):N % Adjust start index to allow for past symbols and feedback
            current_symbol_idx = (i-1)*symbol_spacing_samples + 1 + delay_system;
            
            % Calculate indices for past and future symbols
            past_symbol_indices = current_symbol_idx - (1:num_past_future_symbols) * symbol_spacing_samples;
            future_symbol_indices = current_symbol_idx + (1:num_past_future_symbols) * symbol_spacing_samples;
            
            % Combine all relevant indices for boundary check
            all_relevant_indices = [past_symbol_indices, current_symbol_idx-half_win:current_symbol_idx+half_win, future_symbol_indices];

            % Check if all required samples are within the received signal bounds
            if min(all_relevant_indices) >= 1 && max(all_relevant_indices) <= length(rx)
                win_samples = real(rx(current_symbol_idx-half_win:current_symbol_idx+half_win));
                past_samples = real(rx(past_symbol_indices));
                future_samples = real(rx(future_symbol_indices));
                
                % Concatenate all features: past/future received samples, window samples, and feedback decisions
                features = [past_samples; win_samples; future_samples; decision_history];

                % Predict using the neural network
                probs = predict(net, features');
                [~, pred_class] = max(probs, [], 2);
                pred_bit_nn = pred_class - 1; % Convert 1-based class to 0/1 bit

                % Compare with true bit for NN error count
                if pred_bit_nn ~= bits(i)
                    errors_nn = errors_nn + 1;
                end

                % Simple threshold detector for comparison
                center_sample = real(rx(current_symbol_idx));
                pred_bit_th = (center_sample > 0);
                if pred_bit_th ~= bits(i)
                    errors_threshold = errors_threshold + 1;
                end

                total_bits = total_bits + 1;

                % Update decision history with the *predicted* bit for the next symbol
                decision_history = [pred_bit_nn; decision_history(1:end-1)];

                % Exit conditions for the inner loop
                if errors_nn >= min_err || total_bits >= max_bits
                    return;
                end
            end
        end
    end
end

% Removed the old test_bpsk function as it's not used in this DF context
% You might want to keep it if you need to test non-DF models with this script
% but the current logic always calls test_bpsk_df.