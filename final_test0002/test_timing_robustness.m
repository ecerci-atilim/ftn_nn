% =========================================================================
% FTN Timing Robustness Test
% =========================================================================
% This script demonstrates the "novelty" of the fractional sampling (window)
% approach: ROBUSTNESS against timing jitter/offset.
%
% 1. Trains NN models under PERFECT timing conditions (offset = 0).
% 2. Tests NN models under IMPERFECT timing (offset > 0).
% =========================================================================

clear; clc; close all;

% Parameters
tau = 0.7;
SNR_dB = 10; % Fixed SNR where we see the error floor
offsets = 0 : 0.05 : 0.45; % 0% to 45% symbol period timing error
N_train = 50000;
N_test = 50000;

fprintf('========================================\n');
fprintf('  FTN Timing Robustness Test\n');
fprintf('  tau = %.1f, SNR = %d dB\n', tau, SNR_dB);
fprintf('  Comparing Neighbor (Symbol-Rate) vs Window (Fractional)\n');
fprintf('========================================\n');

%% 1. Generate Training Data (Perfect Timing)
fprintf('\n[1/3] Generating Training Data (Perfect Timing)...\n');

% Generate FTN signal
[tx_bits, ~, rx_mf, ~, step, ~] = generate_ftn_rx_with_offset(tau, N_train, SNR_dB, 0);

% 1A. Neighbor Features (7 samples - symbol rate)
neighbor_offsets = (-3:3) * step;
X_train_Nb = extract_features(rx_mf, step, neighbor_offsets);

% 1B. Window Features (43 samples - fractional)
window_offsets = -21:21;
X_train_Win = extract_features(rx_mf, step, window_offsets);

% --- FIX: Ensure X and Y have same length ---
num_samples = min([size(X_train_Nb, 2), size(X_train_Win, 2), length(tx_bits)]);
X_train_Nb = X_train_Nb(:, 1:num_samples);
X_train_Win = X_train_Win(:, 1:num_samples);
Y_train = categorical(tx_bits(1:num_samples));
% --------------------------------------------

% Normalize inputs
X_train_Nb = (X_train_Nb - mean(X_train_Nb, 2)) ./ std(X_train_Nb, 0, 2);
X_train_Win = (X_train_Win - mean(X_train_Win, 2)) ./ std(X_train_Win, 0, 2);

%% 2. Train Networks
fprintf('[2/3] Training Models...\n');

% Training Options
opts = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 1024, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% Model 1: Neighbor FC
layers_nb = [
    featureInputLayer(7, 'Normalization', 'none')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

fprintf('  Training Neighbor FC... ');
net_nb = trainNetwork(X_train_Nb', Y_train, layers_nb, opts);
fprintf('Done.\n');

% Model 2: Window FC
layers_win = [
    featureInputLayer(43, 'Normalization', 'none')
    fullyConnectedLayer(32) 
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

fprintf('  Training Window FC...   ');
net_win = trainNetwork(X_train_Win', Y_train, layers_win, opts);
fprintf('Done.\n');

%% 3. Test Robustness (Sweep Offsets)
fprintf('\n[3/3] Testing Robustness against Timing Offset...\n');
fprintf('  Offset | Neighbor BER | Window BER\n');
fprintf('  -------|--------------|------------\n');

BER_Nb = zeros(size(offsets));
BER_Win = zeros(size(offsets));

for i = 1:length(offsets)
    off = offsets(i);
    
    % Generate test data WITH OFFSET
    [tx_bits_test, ~, rx_mf_test, ~, ~, ~] = generate_ftn_rx_with_offset(tau, N_test, SNR_dB, off);
    
    % Extract Features
    X_test_Nb = extract_features(rx_mf_test, step, neighbor_offsets);
    X_test_Win = extract_features(rx_mf_test, step, window_offsets);
    
    % --- FIX: Match labels ---
    num_test_samples = min([size(X_test_Nb, 2), size(X_test_Win, 2), length(tx_bits_test)]);
    X_test_Nb = X_test_Nb(:, 1:num_test_samples);
    X_test_Win = X_test_Win(:, 1:num_test_samples);
    Y_test_labels = tx_bits_test(1:num_test_samples);
    % -------------------------
    
    % Normalize
    X_test_Nb = (X_test_Nb - mean(X_test_Nb, 2)) ./ std(X_test_Nb, 0, 2);
    X_test_Win = (X_test_Win - mean(X_test_Win, 2)) ./ std(X_test_Win, 0, 2);
    
    % Predict Neighbor
    Y_pred_Nb = classify(net_nb, X_test_Nb');
    err_nb = sum(double(string(Y_pred_Nb)) ~= double(string(Y_test_labels)));
    BER_Nb(i) = err_nb / num_test_samples;
    
    % Predict Window
    Y_pred_Win = classify(net_win, X_test_Win');
    err_win = sum(double(string(Y_pred_Win)) ~= double(string(Y_test_labels)));
    BER_Win(i) = err_win / num_test_samples;
    
    fprintf('   %.2f  |   %.2e   |   %.2e\n', off, BER_Nb(i), BER_Win(i));
end

%% 4. Plot Results
if exist('figures', 'dir') == 0; mkdir('figures'); end

figure('Position', [100, 100, 800, 600]);
semilogy(offsets, BER_Nb, 'r-o', 'LineWidth', 2, 'DisplayName', 'Neighbor (Symbol-Rate)');
hold on;
semilogy(offsets, BER_Win, 'b-s', 'LineWidth', 2, 'DisplayName', 'Window (Fractional)');
grid on;
xlabel('Timing Offset (\tau / T_{sym})');
ylabel('Bit Error Rate (BER)');
title(sprintf('Robustness to Timing Jitter (Tau=%.1f, SNR=%ddB)', tau, SNR_dB));
legend('Location', 'best');
axis([0 max(offsets) 1e-5 1]);

saveas(gcf, 'figures/timing_robustness.png');
fprintf('\nDone! Plot saved to figures/timing_robustness.png\n');


%% Helper Functions
function [bits, symbols, rx_mf, h, step, sps] = generate_ftn_rx_with_offset(tau, N, SNR_dB, offset_ratio)
    sps = 10; % High resolution
    step = round(tau * sps);
    span = 6; % Pulse span
    beta = 0.3; % Rolloff
    
    % Generate Pulse
    h = rcosdesign(beta, span, sps, 'sqrt');
    
    % Generate Data
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    
    % Transmit (Upsample -> Conv)
    tx_up = upsample(symbols, step);
    tx = conv(tx_up, h);
    
    % Channel (AWGN)
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    noise = sqrt(noise_var) * randn(size(tx));
    rx_noisy = tx + noise;
    
    % Matched Filter
    rx_mf_raw = conv(rx_noisy, h);
    
    % Apply Timing Offset
    offset_samples = round(offset_ratio * sps); 
    
    % Shift signal
    rx_mf = rx_mf_raw;
    if offset_samples > 0
        rx_mf = [zeros(offset_samples,1); rx_mf(1:end-offset_samples)];
    end
end

function X = extract_features(rx_mf, step, offsets)
    span = 6; sps = 10;
    delay = span*sps + 1;
    
    num_syms = floor((length(rx_mf) - delay - max(offsets)) / step);
    
    % Pre-allocate
    X = zeros(length(offsets), num_syms);
    
    for k = 1:num_syms
        center_idx = delay + (k-1)*step;
        indices = center_idx + offsets;
        X(:, k) = rx_mf(indices);
    end
end