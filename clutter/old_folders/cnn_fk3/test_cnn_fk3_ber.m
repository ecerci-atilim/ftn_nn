%% CNN-FK3 BER Test Script
%  Test trained model and compare with M-BCJR
%
%  Author: Emre Cerci
%  Date: January 2026

clear; clc; close all;

%% ========== CONFIGURATION ==========
tau = 0.7;
beta = 0.35;
SNR_range = 0:2:14;

% Monte Carlo parameters
min_errors = 100;
max_symbols = 1e7;
frame_size = 5000;
max_frames = ceil(max_symbols / frame_size);

%% ========== LOAD MODEL ==========
model_files = dir(sprintf('cnn_fk3_tau%.1f_*.mat', tau));

if ~isempty(model_files)
    fprintf('Loading trained model: %s\n', model_files(end).name);
    data = load(model_files(end).name);
    params = data.params;
    config = data.config;
else
    error('No trained model found for tau=%.1f. Run train_cnn_fk3.m first.', tau);
end

N = config.N;
num_fk_layers = config.num_fk_layers;
filters_per_layer = config.filters_per_layer;
total_filters = config.total_filters;
input_size = 2*N + 1;

fprintf('Model loaded: N=%d, Total filters=%d\n\n', N, total_filters);

%% ========== FTN SYSTEM SETUP ==========
sps = 10;
span = 6;
step = round(tau * sps);
delay = span * sps;

h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);

%% ========== BER SIMULATION ==========
fprintf('============================================\n');
fprintf('CNN-FK3 BER Test (tau=%.1f)\n', tau);
fprintf('============================================\n');
fprintf('SNR (dB) | BER       | Errors   | Symbols    | Frames\n');
fprintf('---------|-----------|----------|------------|-------\n');

BER_cnn = zeros(size(SNR_range));
total_errors = zeros(size(SNR_range));
total_symbols = zeros(size(SNR_range));

for snr_idx = 1:length(SNR_range)
    SNR_dB = SNR_range(snr_idx);
    
    EbN0_lin = 10^(SNR_dB / 10);
    N0 = 1 / EbN0_lin;
    noise_std = sqrt(N0 / 2);
    
    errors = 0;
    symbols = 0;
    frame_count = 0;
    
    rng(42);  % Reproducible
    
    while (errors < min_errors) && (frame_count < max_frames)
        frame_count = frame_count + 1;
        
        % Generate random bits
        bits = randi([0 1], frame_size, 1);
        tx_symbols = 2*bits - 1;  % BPSK
        
        % Pad
        symbols_padded = [-ones(N, 1); tx_symbols; -ones(N, 1)];
        
        % Transmit
        tx_up = zeros(length(symbols_padded) * step, 1);
        tx_up(1:step:end) = symbols_padded;
        tx_sig = conv(tx_up, h);
        
        % Add noise
        noise = noise_std * randn(size(tx_sig));
        rx_sig = tx_sig + noise;
        
        % Matched filter
        rx_mf = conv(rx_sig, h);
        
        % Extract windows for CNN
        total_delay = 2 * delay;
        X_test = zeros(input_size, frame_size);
        
        for k = 1:frame_size
            center_idx = total_delay + (N + k - 1) * step + 1;
            
            for w = 1:input_size
                sym_offset = w - (N + 1);
                sample_pos = center_idx + sym_offset * step;
                if sample_pos > 0 && sample_pos <= length(rx_mf)
                    X_test(w, k) = rx_mf(sample_pos);
                end
            end
        end
        
        % CNN prediction
        X_dl = dlarray(single(X_test));
        y_pred = extractdata(forwardPass(params, X_dl, N, num_fk_layers, filters_per_layer));
        bits_hat = double(y_pred > 0.5)';
        
        % Count errors
        errors = errors + sum(bits_hat ~= bits);
        symbols = symbols + frame_size;
    end
    
    BER_cnn(snr_idx) = errors / symbols;
    total_errors(snr_idx) = errors;
    total_symbols(snr_idx) = symbols;
    
    fprintf('  %2d     | %.2e  | %6d   | %10.0f | %d\n', ...
            SNR_dB, BER_cnn(snr_idx), errors, symbols, frame_count);
    
    if BER_cnn(snr_idx) == 0
        fprintf('  (No errors detected)\n');
        break;
    end
end

%% ========== THEORETICAL BPSK ==========
EbN0_lin = 10.^(SNR_range/10);
BER_BPSK = qfunc(sqrt(2*EbN0_lin));

%% ========== PLOT RESULTS ==========
figure('Position', [100 100 800 600]);

semilogy(SNR_range, BER_cnn, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'DisplayName', sprintf('CNN-FK3 (\\tau=%.1f)', tau));
hold on;
semilogy(SNR_range, BER_BPSK, 'k--', 'LineWidth', 1.5, ...
         'DisplayName', 'BPSK (ISI-free)');

% Load M-BCJR results if available
bcjr_file = 'ftn_bcjr_results.mat';
if exist(bcjr_file, 'file')
    bcjr_data = load(bcjr_file);
    semilogy(bcjr_data.SNR_range, bcjr_data.BER_results, 'rs-', 'LineWidth', 2, ...
             'MarkerSize', 8, 'DisplayName', 'M-BCJR');
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('CNN-FK3 vs M-BCJR (\\tau=%.1f, \\beta=%.2f)', tau, beta));
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-6 1]);
xlim([SNR_range(1) SNR_range(end)]);

saveas(gcf, sprintf('ber_cnn_fk3_tau%.1f.png', tau));

%% ========== SAVE RESULTS ==========
save(sprintf('ber_cnn_fk3_results_tau%.1f.mat', tau), ...
     'SNR_range', 'BER_cnn', 'BER_BPSK', 'total_errors', 'total_symbols', ...
     'tau', 'beta', 'N');

fprintf('\nTest completed. Results saved.\n');

%% ========== COMPARISON TABLE ==========
fprintf('\n============================================\n');
fprintf('Performance Comparison @ 10 dB\n');
fprintf('============================================\n');

[~, idx_10dB] = min(abs(SNR_range - 10));

fprintf('BPSK Theoretical: %.2e\n', BER_BPSK(idx_10dB));
fprintf('CNN-FK3:          %.2e\n', BER_cnn(idx_10dB));

if exist(bcjr_file, 'file')
    [~, idx_bcjr] = min(abs(bcjr_data.SNR_range - 10));
    fprintf('M-BCJR:           %.2e\n', bcjr_data.BER_results(idx_bcjr));
end

%% ========== FORWARD PASS FUNCTION ==========
function y = forwardPass(params, X, N, num_fk_layers, filters_per_layer)
    center = N + 1;
    
    % FK layers
    fk_outputs = cell(num_fk_layers, 1);
    for i = 1:num_fk_layers
        idx_left = center - i;
        idx_right = center + i;
        triplet = X([idx_left, center, idx_right], :);
        
        W = params.(sprintf('W_fk%d', i));
        b = params.(sprintf('b_fk%d', i));
        
        z = W' * triplet + b;
        fk_outputs{i} = tanh(z);
    end
    
    % Concatenate
    concat_out = cat(1, fk_outputs{:});
    
    % Dense
    z_dense = params.W_dense' * concat_out + params.b_dense;
    h_dense = tanh(z_dense);
    
    % Output
    z_out = params.W_out' * h_dense + params.b_out;
    y = sigmoid(z_out);
end