% çalışıyor

clear; clc; close all;
set(groot,'defaultAxesTickLabelInterpreter','latex');      % Interpreter definition for axes ticks of figures
set(groot,'defaulttextinterpreter','latex');               % Interpreter definition for default strings casted on figures
set(groot,'defaultLegendInterpreter','latex');             % Interpreter definitions for default legend strings displayed on figures

[file, path] = uigetfile('mat/bpsk_simple/*.mat');
if isequal(file, 0), error('No file selected'); end
load(fullfile(path, file)); % This will load net, tau, window_len, input_size, and num_past_future_symbols

fprintf('Testing BPSK Equalizer (no DF): tau=%.1f\n\n', tau);
fprintf('Model trained with %d past and %d future symbols as features.\n', num_past_future_symbols, num_past_future_symbols);


SNR_range = 0:2:18;
ber_nn = zeros(size(SNR_range));
ber_threshold = zeros(size(SNR_range));
ber_theory_awgn = qfunc(sqrt(2 * 10.^(SNR_range/10)));

min_errors = 500;
max_bits = 1e7;

for idx = 1:length(SNR_range)
    snr = SNR_range(idx);
    % Pass num_past_future_symbols to the test_bpsk function
    [errors_nn, total, errors_threshold] = test_bpsk(net, tau, snr, window_len, num_past_future_symbols, min_errors, max_bits);
    ber_nn(idx) = errors_nn / total;
    ber_threshold(idx) = errors_threshold / total;
    fprintf('SNR=%2d dB: NN=%.4e, Threshold=%.4e (%d/%d errors)\n', ...
        snr, ber_nn(idx), ber_threshold(idx), errors_nn, total);
end

figure('Position', [100 100 800 500]);

semilogy(SNR_range, ber_theory_awgn, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory (AWGN)');
hold on;
semilogy(SNR_range, ber_threshold, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Uncoded FTN');
semilogy(SNR_range, ber_nn, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('NN (τ=%.1f)', tau)); % Updated legend

grid on;
xlabel('$E_b$/$N_0$ (dB)');
ylabel('BER');
legend('Location', 'southwest');

% if abs(tau - 1.0) < 0.01
%     title('BPSK τ=1.0 (Nyquist - No ISI) with Extended Features');
% else
%     title(sprintf('BPSK Equalization (τ=%.1f with ISI) with Extended Features', tau));
% end

ylim([1e-6 0.5]);

if ~exist(fullfile(path, 'results'), 'dir'), mkdir(fullfile(path, 'results')); end
save(fullfile(path, 'results', strrep(file, '.mat', '_results.mat')), 'SNR_range', 'ber_nn', 'ber_threshold', 'ber_theory_awgn');

function [errors_nn, total_bits, errors_threshold] = test_bpsk(net, tau, SNR_dB, win_len, num_past_future_symbols, min_err, max_bits)
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
    
    N = 100000;
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
        % Current symbol's sampling point index in the received signal
        current_symbol_idx = (i-1)*symbol_spacing_samples + 1 + delay_system;
        
        % Calculate indices for past and future symbols
        past_symbol_indices = current_symbol_idx - (1:num_past_future_symbols) * symbol_spacing_samples;
        future_symbol_indices = current_symbol_idx + (1:num_past_future_symbols) * symbol_spacing_samples;
        
        all_relevant_indices = [past_symbol_indices, current_symbol_idx-half_win:current_symbol_idx+half_win, future_symbol_indices];
        
        % Check if all relevant indices are within the bounds of rx
        if min(all_relevant_indices) >= 1 && max(all_relevant_indices) <= length(rx)
            % Extract window samples around the current symbol
            win_samples = real(rx(current_symbol_idx-half_win:current_symbol_idx+half_win));
            
            % Extract past symbol samples
            past_samples = real(rx(past_symbol_indices));
            
            % Extract future symbol samples
            future_samples = real(rx(future_symbol_indices));
            
            % Concatenate all features
            features = [past_samples; win_samples; future_samples];
            
            feature_buffer = [feature_buffer; features(:)']; % Ensure it's a row vector
            bit_buffer = [bit_buffer; bits(i)];
            center_buffer = [center_buffer; real(rx(current_symbol_idx))]; % Still use only the center for threshold
            
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