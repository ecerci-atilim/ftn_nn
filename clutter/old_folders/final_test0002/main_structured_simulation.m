clear; clc; close all;

% Parameters
tau = 0.7;
SNR_dB = 10; % Critical region
N_train = 50000;
N_test = 50000;

fprintf('========================================\n');
fprintf('  FTN STRUCTURED CNN SIMULATION\n');
fprintf('  Tau=%.1f, SNR=%ddB\n', tau, SNR_dB);
fprintf('========================================\n');

%% 1. Generate Data
fprintf('[1/4] Generating Data...\n');
[tx_bits_train, ~, rx_mf_train, ~, step, sps] = generate_ftn_data(tau, N_train, SNR_dB);
[tx_bits_test, ~, rx_mf_test, ~, ~, ~] = generate_ftn_data(tau, N_test, SNR_dB);

%% 2. Extract Features

% A. Neighbor Features (Baseline)
fprintf('[2/4] Extracting Neighbor Features...\n');
offsets_nb = (-3:3) * step;
X_train_Nb = extract_simple_features(rx_mf_train, step, offsets_nb);
X_test_Nb = extract_simple_features(rx_mf_test, step, offsets_nb);

% B. STRUCTURED Features (The Novelty)
fprintf('[2/4] Extracting STRUCTURED Features...\n');
X_train_Struct = extract_structured_matrix(rx_mf_train, step, sps);
X_test_Struct = extract_structured_matrix(rx_mf_test, step, sps);

% --- FIX: Align all dimensions safely ---
% Train Alignment
n_train_limit = min([size(X_train_Nb, 2), size(X_train_Struct, 4), length(tx_bits_train)]);
X_train_Nb = X_train_Nb(:, 1:n_train_limit);
X_train_Struct = X_train_Struct(:, :, 1, 1:n_train_limit);
Y_train = categorical(tx_bits_train(1:n_train_limit));

% Test Alignment
n_test_limit = min([size(X_test_Nb, 2), size(X_test_Struct, 4), length(tx_bits_test)]);
X_test_Nb = X_test_Nb(:, 1:n_test_limit);
X_test_Struct = X_test_Struct(:, :, 1, 1:n_test_limit);
Y_test_labels = tx_bits_test(1:n_test_limit);
% ----------------------------------------

% Normalize
X_train_Nb = (X_train_Nb - mean(X_train_Nb,2)) ./ std(X_train_Nb,0,2);
X_test_Nb = (X_test_Nb - mean(X_test_Nb,2)) ./ std(X_test_Nb,0,2);
% Global normalization for structured input
mu_struct = mean(X_train_Struct, 'all');
sig_struct = std(X_train_Struct, 0, 'all');
X_train_Struct = (X_train_Struct - mu_struct) ./ sig_struct;
X_test_Struct = (X_test_Struct - mu_struct) ./ sig_struct;

%% 3. Train Models

% Options
opts = trainingOptions('adam', ...
    'MaxEpochs', 20, ... % Increased epochs for complex structure
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% --- Model A: Neighbor FC (Baseline) ---
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

% --- Model B: Structured CNN ---
% Input: 7x7x1 Image
% Rows: Neighbor symbols (-3 to +3)
% Cols: Samples around that symbol
layers_struct = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    
    % Conv Layer 1: Look at each symbol row independently
    % Filter [1 7]: Scans horizontally (across samples of ONE symbol)
    % Padding 'same' keeps width 7
    convolution2dLayer([1 7], 32, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    
    % Conv Layer 2: Combine info across neighbor symbols (Inter-Symbol Interference)
    % Filter [7 1]: Scans vertically (combines all 7 rows into 1)
    % Padding 0 (valid) reduces height 7 -> 1
    convolution2dLayer([7 1], 16, 'Padding', 0)
    batchNormalizationLayer
    reluLayer
    
    % Output is now 1x7x16 features (flat enough for FC)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

fprintf('  Training Structured CNN... ');
net_struct = trainNetwork(X_train_Struct, Y_train, layers_struct, opts);
fprintf('Done.\n');

%% 4. Test and Compare
fprintf('\n[4/4] Testing Performance...\n');

% Predict Neighbor
Y_pred_Nb = classify(net_nb, X_test_Nb');
err_nb = sum(double(string(Y_pred_Nb)) ~= double(string(Y_test_labels)));
ber_nb = err_nb / n_test_limit;

% Predict Structured
Y_pred_Struct = classify(net_struct, X_test_Struct);
err_struct = sum(double(string(Y_pred_Struct)) ~= double(string(Y_test_labels)));
ber_struct = err_struct / n_test_limit;

fprintf('================ RESULTS ================\n');
fprintf('Method           | Input Structure | BER\n');
fprintf('-----------------|-----------------|--------\n');
fprintf('Neighbor FC      | 7 instants      | %.2e\n', ber_nb);
fprintf('Structured CNN   | 7x7 matrix      | %.2e\n', ber_struct);
fprintf('-----------------------------------------\n');

if ber_struct < ber_nb
    fprintf('SUCCESS: Structured approach is BETTER!\n');
    fprintf('Improvement: %.2f%%\n', (1 - ber_struct/ber_nb) * 100);
else
    fprintf('RESULT: Structured approach is similar/worse.\n');
end


%% Helper Functions
function [bits, symbols, rx_mf, h, step, sps] = generate_ftn_data(tau, N, SNR_dB)
    sps = 10;
    step = round(tau * sps);
    span = 6;
    beta = 0.3;
    h = rcosdesign(beta, span, sps, 'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    tx = conv(upsample(symbols, step), h);
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
    rx_mf = conv(rx_noisy, h);
    
    rx_mf = rx_mf / std(rx_mf);
end

function X = extract_simple_features(rx_mf, step, offsets)
    span = 6; sps = 10;
    delay = span*sps + 1;
    num_syms = floor((length(rx_mf) - delay - max(offsets)) / step);
    X = zeros(length(offsets), num_syms);
    for k = 1:num_syms
        center_idx = delay + (k-1)*step;
        indices = center_idx + offsets;
        X(:, k) = rx_mf(indices);
    end
end

function X_struct = extract_structured_matrix(rx_mf, step, sps)
    % Extracts a 7x7x1xN 4D array for CNN input
    span = 6;
    delay = span*sps + 1;
    local_window = -3:3; % 7 samples wide
    symbol_positions = -3:3; % 7 symbols
    
    max_offset = max(symbol_positions)*step + max(local_window);
    num_syms = floor((length(rx_mf) - delay - max_offset) / step);
    
    X_struct = zeros(7, 7, 1, num_syms);
    
    for k = 1:num_syms
        current_center = delay + (k-1)*step;
        for r = 1:7 
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos*step;
            indices = neighbor_center + local_window;
            X_struct(r, :, 1, k) = rx_mf(indices);
        end
    end
end