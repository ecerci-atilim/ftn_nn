% =========================================================================
% FTN DETECTION: Proposed Structured CNN vs Conventional Neighbor
% =========================================================================
% Comparison of Bit Error Rate (BER) performance across SNR range.
%
% Method 1: Neighbor (Symbol-Rate) -> Feedforward NN
% Method 2: Proposed Structured Input (Fractional) -> CNN
%           (Uses 7x7 interference-aware input matrix)
% =========================================================================

clear; clc; close all;

% --- Simulation Parameters ---
tau = 0.7;
Train_SNR = 10;       % Train at a specific SNR where patterns are visible
Test_SNR_range = 0:2:14; 
N_train = 50000;
N_test_per_point = 50000; 

fprintf('================================================\n');
fprintf('  FTN FINAL COMPARISON: BER vs SNR\n');
fprintf('  Tau = %.1f\n', tau);
fprintf('  Training SNR = %d dB\n', Train_SNR);
fprintf('================================================\n');

%% 1. TRAINING PHASE
fprintf('\n[1/3] Training Models (at %d dB)...\n', Train_SNR);

% Generate Training Data
[tx_bits_train, ~, rx_mf_train, ~, step, sps] = generate_ftn_data(tau, N_train, Train_SNR);

% --- A. Prepare Neighbor Features ---
offsets_nb = (-3:3) * step;
X_train_Nb = extract_simple_features(rx_mf_train, step, offsets_nb);

% --- B. Prepare Structured Features ---
X_train_Struct = extract_structured_matrix(rx_mf_train, step, sps);

% --- Align Dimensions ---
n_lim = min([size(X_train_Nb, 2), size(X_train_Struct, 4), length(tx_bits_train)]);
X_train_Nb = X_train_Nb(:, 1:n_lim);
X_train_Struct = X_train_Struct(:, :, 1, 1:n_lim);
Y_train = categorical(tx_bits_train(1:n_lim));

% --- Normalize ---
% Neighbor: Row-wise normalization
mean_nb = mean(X_train_Nb, 2);
std_nb = std(X_train_Nb, 0, 2);
X_train_Nb = (X_train_Nb - mean_nb) ./ std_nb;

% Structured: Global normalization
mean_struct = mean(X_train_Struct, 'all');
std_struct = std(X_train_Struct, 0, 'all');
X_train_Struct = (X_train_Struct - mean_struct) ./ std_struct;

% --- Train Neighbor Network ---
opts = trainingOptions('adam', 'MaxEpochs', 15, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 1e-3, 'Shuffle', 'every-epoch', 'Verbose', false);

% FIX: Use semicolons to create a clean column vector of layers
layers_nb = [
    featureInputLayer(7, 'Normalization', 'none');
    fullyConnectedLayer(32);
    reluLayer;
    fullyConnectedLayer(16);
    reluLayer;
    fullyConnectedLayer(2);
    softmaxLayer;
    classificationLayer];

fprintf('  Training Neighbor FC... ');
net_nb = trainNetwork(X_train_Nb', Y_train, layers_nb, opts);
fprintf('Done.\n');

% --- Train Structured CNN ---
% FIX: Use semicolons here as well
layers_struct = [
    imageInputLayer([7 7 1], 'Normalization', 'none');
    
    % Conv1: Scan samples within each symbol (Horizontal)
    convolution2dLayer([1 7], 32, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer;
    
    % Conv2: Combine across interfering symbols (Vertical)
    convolution2dLayer([7 1], 16, 'Padding', 0);
    batchNormalizationLayer;
    reluLayer;
    
    fullyConnectedLayer(2);
    softmaxLayer;
    classificationLayer];

fprintf('  Training Structured CNN... ');
net_struct = trainNetwork(X_train_Struct, Y_train, layers_struct, opts);
fprintf('Done.\n');


%% 2. TESTING PHASE
fprintf('\n[2/3] Testing across SNR range...\n');
BER_Nb = zeros(size(Test_SNR_range));
BER_Struct = zeros(size(Test_SNR_range));

fprintf('  SNR | Neighbor | Structured | Improvement\n');
fprintf('  ----|----------|------------|------------\n');

for i = 1:length(Test_SNR_range)
    snr = Test_SNR_range(i);
    
    % Generate Test Data
    [tx_bits_test, ~, rx_mf_test, ~, ~, ~] = generate_ftn_data(tau, N_test_per_point, snr);
    
    % Extract
    X_test_Nb = extract_simple_features(rx_mf_test, step, offsets_nb);
    X_test_Struct = extract_structured_matrix(rx_mf_test, step, sps);
    
    % Align
    n_lim_test = min([size(X_test_Nb, 2), size(X_test_Struct, 4), length(tx_bits_test)]);
    X_test_Nb = X_test_Nb(:, 1:n_lim_test);
    X_test_Struct = X_test_Struct(:, :, 1, 1:n_lim_test);
    Y_test = tx_bits_test(1:n_lim_test);
    
    % Normalize (Using Training Stats!)
    X_test_Nb = (X_test_Nb - mean_nb) ./ std_nb;
    X_test_Struct = (X_test_Struct - mean_struct) ./ std_struct;
    
    % Predict Neighbor
    Y_pred_Nb = classify(net_nb, X_test_Nb');
    err_nb = sum(double(string(Y_pred_Nb)) ~= double(string(Y_test)));
    BER_Nb(i) = err_nb / n_lim_test;
    
    % Predict Structured
    Y_pred_Struct = classify(net_struct, X_test_Struct);
    err_struct = sum(double(string(Y_pred_Struct)) ~= double(string(Y_test)));
    BER_Struct(i) = err_struct / n_lim_test;
    
    imp = (1 - BER_Struct(i)/BER_Nb(i))*100;
    if isnan(imp), imp=0; end
    fprintf('   %2d | %.2e | %.2e   | %5.1f%%\n', snr, BER_Nb(i), BER_Struct(i), imp);
end

%% 3. PLOTTING
fprintf('\n[3/3] Generating Plot...\n');
if exist('figures', 'dir') == 0; mkdir('figures'); end

figure('Position', [100, 100, 800, 600]);
semilogy(Test_SNR_range, BER_Nb, 'r-o', 'LineWidth', 2, 'DisplayName', 'Conventional Neighbor (Symbol-Rate)');
hold on;
semilogy(Test_SNR_range, BER_Struct, 'b-s', 'LineWidth', 2, 'DisplayName', 'Proposed Structured CNN (Fractional)');
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title(sprintf('Performance Comparison (Tau=%.1f)', tau));
legend('Location', 'southwest');
axis([0 14 1e-5 1]);

saveas(gcf, 'figures/final_ber_comparison.png');
fprintf('Done! Results saved to "figures/final_ber_comparison.png"\n');


%% --- Helper Functions ---
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
    rx_mf = rx_mf / std(rx_mf); % Normalize
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
    span = 6;
    delay = span*sps + 1;
    local_window = -3:3; 
    symbol_positions = -3:3;
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