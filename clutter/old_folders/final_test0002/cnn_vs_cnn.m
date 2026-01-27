% =========================================================================
% FTN ULTIMATE SHOWDOWN: 1D CNN (Neighbor) vs 2D CNN (Structured)
% =========================================================================
% "Is it the Architecture or the Sampling?"
%
% To answer this definitively, we apply CONVOLUTIONAL NETWORKS to BOTH.
%
% Method A: Neighbor (Symbol-Rate) -> 1D CNN
%   - Input: 7x1 Sequence
%   - Architecture: Conv1D -> BatchNorm -> ReLU -> FC
%   - Checks if Deep Learning can squeeze more juice out of 7 samples.
%
% Method B: Structured (Fractional) -> 2D CNN
%   - Input: 7x7 Image
%   - Architecture: Conv2D -> BatchNorm -> ReLU -> FC
%
% If Method B wins, it proves that FRACTIONAL SAMPLES contain information
% that Symbol-Rate samples simply DO NOT HAVE, regardless of the model.
% =========================================================================

clear; clc; close all;

% --- Simulation Parameters ---
tau = 0.7;
Train_SNR = 10;
Test_SNR_range = 0:2:14; 
N_train = 50000;
N_test = 50000;

fprintf('================================================\n');
fprintf('  ULTIMATE FAIR TEST: CNN vs CNN\n');
fprintf('  Does Deep Learning fix Neighbor''s error floor?\n');
fprintf('================================================\n');

%% 1. Data Generation
[tx, ~, rx, ~, step, sps] = generate_ftn_data(tau, N_train, Train_SNR);

% --- A. Neighbor Features (Sequence for 1D CNN) ---
offsets_nb = (-3:3) * step;
X_nb = extract_simple_features(rx, step, offsets_nb); 
% Reshape for 1D CNN: [Height, Width, Channels, Batch] -> [7, 1, 1, N]
X_nb_cnn = reshape(X_nb, [7, 1, 1, size(X_nb, 2)]);

% --- B. Structured Features (Image for 2D CNN) ---
X_struct = extract_structured_matrix(rx, step, sps);

% --- Alignment ---
lim = min([size(X_nb, 2), size(X_struct, 4), length(tx)]);
X_nb_cnn = X_nb_cnn(:, :, 1, 1:lim);
X_struct = X_struct(:, :, 1, 1:lim);
Y = categorical(tx(1:lim));

% --- Normalization ---
mu_n = mean(X_nb_cnn, 'all'); sig_n = std(X_nb_cnn, 0, 'all');
X_nb_cnn = (X_nb_cnn - mu_n) ./ sig_n;

mu_s = mean(X_struct, 'all'); sig_s = std(X_struct, 0, 'all');
X_struct = (X_struct - mu_s) ./ sig_s;

%% 2. Models Definition (The Fairness Part)

opts = trainingOptions('adam', 'MaxEpochs', 20, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 1e-3, 'Shuffle', 'every-epoch', 'Verbose', false);

% --- Model A: Neighbor 1D CNN ---
% Trying to be as smart as possible with 7 samples
layers_1d = [
    imageInputLayer([7 1 1], 'Normalization', 'none') % Treat as 7x1 image
    
    % Conv1D equivalent: Scan over the 7 samples (Kernel size 3x1)
    convolution2dLayer([3 1], 32, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    
    % Another Conv to deepen features
    convolution2dLayer([3 1], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

fprintf('Training Neighbor 1D-CNN... ');
net_1d = trainNetwork(X_nb_cnn, Y, layers_1d, opts);
fprintf('Done.\n');

% --- Model B: Structured 2D CNN (Our Proposed) ---
layers_2d = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    
    % Conv2D: Spatial processing (Horizontal scan)
    convolution2dLayer([1 7], 32, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    
    % Vertical scan (Combine symbols)
    convolution2dLayer([7 1], 16, 'Padding', 0)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

fprintf('Training Structured 2D-CNN... ');
net_2d = trainNetwork(X_struct, Y, layers_2d, opts);
fprintf('Done.\n');

%% 3. Test Loop
fprintf('\nComparing CNN Performance...\n');
fprintf('SNR | Neighbor (1D CNN) | Structured (2D CNN) | Winner\n');
fprintf('----|-------------------|---------------------|--------\n');

BER_1D = zeros(size(Test_SNR_range));
BER_2D = zeros(size(Test_SNR_range));

for i = 1:length(Test_SNR_range)
    snr = Test_SNR_range(i);
    [tx_t, ~, rx_t, ~, ~, ~] = generate_ftn_data(tau, N_test, snr);
    
    % Process Test Data
    Xn_t = extract_simple_features(rx_t, step, offsets_nb);
    Xn_t = reshape(Xn_t, [7, 1, 1, size(Xn_t, 2)]);
    Xs_t = extract_structured_matrix(rx_t, step, sps);
    
    lim_t = min([size(Xn_t, 4), size(Xs_t, 4), length(tx_t)]);
    Xn_t = Xn_t(:, :, 1, 1:lim_t); Xs_t = Xs_t(:, :, 1, 1:lim_t); Y_t = tx_t(1:lim_t);
    
    % Normalize
    Xn_t = (Xn_t - mu_n) ./ sig_n;
    Xs_t = (Xs_t - mu_s) ./ sig_s;
    
    % Predict
    P_1D = classify(net_1d, Xn_t);
    BER_1D(i) = mean(double(string(P_1D)) ~= double(string(Y_t)));
    
    P_2D = classify(net_2d, Xs_t);
    BER_2D(i) = mean(double(string(P_2D)) ~= double(string(Y_t)));
    
    win = "Neighbor";
    if BER_2D(i) < BER_1D(i), win = "Structured"; end
    
    fprintf(' %2d |     %.2e      |      %.2e       | %s\n', snr, BER_1D(i), BER_2D(i), win);
end

%% 4. Plot
if exist('figures', 'dir') == 0; mkdir('figures'); end
figure('Position', [100, 100, 800, 600]);
semilogy(Test_SNR_range, BER_1D, 'r--o', 'LineWidth', 2, 'DisplayName', 'Neighbor (1D CNN)');
hold on;
semilogy(Test_SNR_range, BER_2D, 'b-s', 'LineWidth', 2, 'DisplayName', 'Structured (2D CNN)');
grid on; xlabel('SNR (dB)'); ylabel('BER'); legend('Location', 'southwest');
title(sprintf('Ultimate CNN vs CNN Test (Tau=%.1f)', tau));
saveas(gcf, 'figures/ultimate_cnn_test.png');

%% Helpers
function [bits, symbols, rx_mf, h, step, sps] = generate_ftn_data(tau, N, SNR_dB)
    sps = 10; step = round(tau * sps);
    h = rcosdesign(0.3, 6, sps, 'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    tx = conv(upsample(symbols, step), h);
    
    % Channel
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    
    rx_noisy = tx + sqrt(noise_var)*randn(size(tx));
    rx_mf_raw = conv(rx_noisy, h);
    rx_mf = rx_mf_raw / std(rx_mf_raw); % Normalized MF output
end

function X = extract_simple_features(rx, step, off)
    del = 61; num = floor((length(rx)-del-max(off))/step);
    X = zeros(length(off), num);
    for k=1:num, X(:,k) = rx(del+(k-1)*step+off); end
end

function X = extract_structured_matrix(rx, step, sps)
    del = 61; loc = -3:3; syms = -3:3;
    max_off = max(syms)*step + max(loc);
    num = floor((length(rx)-del-max_off)/step);
    X = zeros(7,7,1,num);
    for k=1:num
        cen = del+(k-1)*step;
        for r=1:7, X(r,:,1,k) = rx(cen+syms(r)*step+loc); end
    end
end