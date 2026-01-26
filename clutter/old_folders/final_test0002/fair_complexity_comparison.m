% =========================================================================
% FTN FAIR COMPARISON: Complexity-Matched Neighbor vs Structured CNN
% =========================================================================
% Hypothesis: The performance gap is due to INFORMATION (sampling), 
% not MODEL CAPACITY (parameters).
%
% To prove this, we give the Neighbor model MORE parameters than the CNN.
%
% Model A: "Giant" Neighbor FC (Symbol-Rate)
%   - Layers: [128, 64]
%   - Parameters: ~9,200
%
% Model B: Proposed Structured CNN (Fractional)
%   - Layers: Conv(32) -> Conv(16)
%   - Parameters: ~3,800
%
% If Model B wins despite having 2.5x FEWER parameters, 
% the fractional sampling advantage is scientifically proven.
% =========================================================================

clear; clc; close all;

% --- Parameters ---
tau = 0.7;
Train_SNR = 10;
Test_SNR_range = 0:2:14;
N_train = 50000;
N_test = 50000;

fprintf('================================================\n');
fprintf('  FTN FAIR COMPLEXITY TEST\n');
fprintf('  Neighbor (High Complexity) vs Structured CNN (Low Complexity)\n');
fprintf('================================================\n');

%% 1. Prepare Data
[tx, ~, rx, ~, step, sps] = generate_ftn_data(tau, N_train, Train_SNR);

% Neighbor Features
offsets_nb = (-3:3) * step;
X_nb = extract_simple_features(rx, step, offsets_nb);

% Structured Features
X_struct = extract_structured_matrix(rx, step, sps);

% Align
lim = min([size(X_nb, 2), size(X_struct, 4), length(tx)]);
X_nb = X_nb(:, 1:lim);
X_struct = X_struct(:, :, 1, 1:lim);
Y = categorical(tx(1:lim));

% Normalize
X_nb = (X_nb - mean(X_nb, 2)) ./ std(X_nb, 0, 2);
mu_s = mean(X_struct, 'all'); sig_s = std(X_struct, 0, 'all');
X_struct = (X_struct - mu_s) ./ sig_s;

%% 2. Models Definition

% Options
opts = trainingOptions('adam', 'MaxEpochs', 15, 'MiniBatchSize', 512, ...
    'InitialLearnRate', 1e-3, 'Shuffle', 'every-epoch', 'Verbose', false);

% --- A. GIANT NEIGHBOR MODEL (~9,200 params) ---
layers_giant = [
    featureInputLayer(7, 'Normalization', 'none');
    fullyConnectedLayer(128); % Huge layer
    reluLayer;
    fullyConnectedLayer(64);  % Big layer
    reluLayer;
    fullyConnectedLayer(2);
    softmaxLayer;
    classificationLayer];

fprintf('Training Giant Neighbor FC (~9k params)... ');
net_giant = trainNetwork(X_nb', Y, layers_giant, opts);
fprintf('Done.\n');

% --- B. COMPACT STRUCTURED CNN (~3,800 params) ---
layers_cnn = [
    imageInputLayer([7 7 1], 'Normalization', 'none');
    convolution2dLayer([1 7], 32, 'Padding', 'same'); % 224 params
    batchNormalizationLayer; reluLayer;
    convolution2dLayer([7 1], 16, 'Padding', 0);      % 3584 params
    batchNormalizationLayer; reluLayer;
    fullyConnectedLayer(2);                           % 34 params
    softmaxLayer; classificationLayer];

fprintf('Training Structured CNN (~3.8k params)... ');
net_cnn = trainNetwork(X_struct, Y, layers_cnn, opts);
fprintf('Done.\n');

%% 3. Test Loop
fprintf('\nComparing Performance across SNR...\n');
fprintf('SNR | Giant Neighbor | Compact CNN | Result\n');
fprintf('----|----------------|-------------|--------\n');

BER_G = zeros(size(Test_SNR_range));
BER_C = zeros(size(Test_SNR_range));

for i = 1:length(Test_SNR_range)
    snr = Test_SNR_range(i);
    [tx_t, ~, rx_t, ~, ~, ~] = generate_ftn_data(tau, N_test, snr);
    
    % Extract & Align
    Xn_t = extract_simple_features(rx_t, step, offsets_nb);
    Xs_t = extract_structured_matrix(rx_t, step, sps);
    lim_t = min([size(Xn_t, 2), size(Xs_t, 4), length(tx_t)]);
    Xn_t = Xn_t(:, 1:lim_t); Xs_t = Xs_t(:, :, 1, 1:lim_t); Y_t = tx_t(1:lim_t);
    
    % Normalize
    Xn_t = (Xn_t - mean(X_nb,2)) ./ std(X_nb,0,2); % Use train stats roughly
    Xs_t = (Xs_t - mu_s) ./ sig_s;
    
    % Predict
    P_G = classify(net_giant, Xn_t');
    BER_G(i) = mean(double(string(P_G)) ~= double(string(Y_t)));
    
    P_C = classify(net_cnn, Xs_t);
    BER_C(i) = mean(double(string(P_C)) ~= double(string(Y_t)));
    
    res = "Neighbor Wins";
    if BER_C(i) < BER_G(i), res = "CNN Wins"; end
    fprintf(' %2d |   %.2e     |   %.2e   | %s\n', snr, BER_G(i), BER_C(i), res);
end

%% 4. Plot
if exist('figures', 'dir') == 0; mkdir('figures'); end
figure('Position', [100, 100, 800, 600]);
semilogy(Test_SNR_range, BER_G, 'r-o', 'LineWidth', 2, 'DisplayName', 'Giant Neighbor FC (~9k param)');
hold on;
semilogy(Test_SNR_range, BER_C, 'b-s', 'LineWidth', 2, 'DisplayName', 'Proposed CNN (~3.8k param)');
grid on; xlabel('SNR (dB)'); ylabel('BER'); legend('Location', 'southwest');
title(sprintf('Fair Complexity Test (Tau=%.1f)', tau));
saveas(gcf, 'figures/fair_complexity_test.png');


%% Helpers
function [bits, symbols, rx_mf, h, step, sps] = generate_ftn_data(tau, N, SNR_dB)
    sps = 10; 
    step = round(tau * sps);
    h = rcosdesign(0.3, 6, sps, 'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1; % Fixed: Assign symbols
    tx = conv(upsample(symbols, step), h);
    no = 1/(2*10^(SNR_dB/10));
    rx_noisy = tx + sqrt(no)*randn(size(tx));
    rx_mf_raw = conv(rx_noisy, h);
    rx_mf = rx_mf_raw / std(rx_mf_raw); % Fixed: Assign output
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