%% FTN_WITH_PA_SATURATION - FTN Detection with PA Saturation & NN Equalization
%
% Compares multiple detection approaches for FTN signaling with PA saturation:
%   1. Threshold (baseline)
%   2. Neighbor NN (symbol-rate sampling)
%   3. Fractional NN (T/2 or denser sampling - captures ISI better)
%   4. Structured CNN (7x7 matrix representation)
%
% Key Insight:
%   Fractional sampling captures more ISI information than symbol-rate sampling,
%   which allows the NN to better learn the interference pattern and achieve
%   superior BER performance, especially with PA nonlinearity.
%
% Author: Emre Cerci
% Date: January 2026

clear; close all; clc;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% FTN Parameters
tau = 0.7;                  % FTN compression factor
beta = 0.3;                 % SRRC roll-off
sps = 10;                   % Samples per symbol
span = 6;                   % Pulse span in symbols

% PA Parameters
PA_MODEL = 'rapp';          % 'rapp', 'saleh', 'soft_limiter'
IBO_dB = 3;                 % Input Back-Off (dB)
PA_ENABLED = true;          % Enable PA saturation

% Simulation Parameters
SNR_train = 10;             % Training SNR (dB)
SNR_test = 0:2:14;          % Test SNR range
N_train = 80000;            % Training symbols
N_test = 30000;             % Test symbols per SNR

% NN Parameters
hidden_sizes = [64, 32];    % Hidden layer sizes
max_epochs = 40;            % Training epochs
mini_batch = 512;           % Mini-batch size

fprintf('========================================\n');
fprintf('FTN with PA Saturation - NN Detection\n');
fprintf('========================================\n');
fprintf('tau = %.2f, beta = %.2f, sps = %d\n', tau, beta, sps);
fprintf('PA Model: %s, IBO = %d dB, Enabled = %d\n', PA_MODEL, IBO_dB, PA_ENABLED);
fprintf('Training: %d symbols @ %d dB\n', N_train, SNR_train);
fprintf('========================================\n\n');

%% ========================================================================
%% PULSE SHAPING & PA SETUP
%% ========================================================================

% SRRC filter (matches reference)
h_srrc = rcosdesign(beta, span, sps, 'sqrt');
h_srrc = h_srrc / norm(h_srrc);
delay = span * sps;
step = round(tau * sps);

% PA parameters
IBO_lin = 10^(IBO_dB/10);
switch lower(PA_MODEL)
    case 'rapp'
        pa_params.G = 1;
        pa_params.Asat = sqrt(IBO_lin);
        pa_params.p = 2;
    case 'saleh'
        pa_params.alpha_a = 2.0;
        pa_params.beta_a = 1.0 / IBO_lin;
        pa_params.alpha_p = pi/3;
        pa_params.beta_p = 1.0 / IBO_lin;
    case 'soft_limiter'
        pa_params.A_lin = sqrt(IBO_lin) * 0.7;
        pa_params.A_sat = sqrt(IBO_lin);
        pa_params.compress = 0.1;
end

%% ========================================================================
%% COMPUTE SAMPLE OFFSETS
%% ========================================================================

% Neighbor: 7 samples at symbol rate (-3T to +3T)
offsets.neighbor = (-3:3) * step;

% Fractional: 7 samples at T/2 spacing (captures more ISI)
frac_step = round(step / 2);
offsets.fractional = (-3:3) * frac_step;

% Dense Fractional: 15 samples at T/7 spacing (maximum ISI capture)
dense_step = max(1, round(step / 7));
offsets.dense = (-7:7) * dense_step;

fprintf('Sample offsets (step=%d):\n', step);
fprintf('  Neighbor (7):    [%s]\n', num2str(offsets.neighbor));
fprintf('  Fractional (7):  [%s]\n', num2str(offsets.fractional));
fprintf('  Dense (15):      [%s]\n\n', num2str(offsets.dense));

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/3] Generating training data...\n');
rng(42);
bits_train = randi([0 1], 1, N_train);
[rx_train, sym_idx_train] = generate_ftn_rx(bits_train, tau, sps, h_srrc, delay, ...
    SNR_train, PA_ENABLED, PA_MODEL, pa_params);

%% ========================================================================
%% TRAIN NEURAL NETWORKS
%% ========================================================================

fprintf('\n[2/3] Training neural networks...\n');

networks = struct();
norm_params = struct();

% 1. Neighbor NN (7 inputs)
fprintf('  Training Neighbor NN (7 inputs)... ');
tic;
[X_nb, y_nb] = extract_features(rx_train, bits_train, sym_idx_train, offsets.neighbor);
[X_nb_norm, mu, sig] = normalize_features(X_nb);
norm_params.neighbor.mu = mu;
norm_params.neighbor.sig = sig;
networks.neighbor = train_nn(X_nb_norm, y_nb, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 2. Fractional NN (7 inputs, T/2 spacing)
fprintf('  Training Fractional NN (7 inputs)... ');
tic;
[X_frac, y_frac] = extract_features(rx_train, bits_train, sym_idx_train, offsets.fractional);
[X_frac_norm, mu, sig] = normalize_features(X_frac);
norm_params.fractional.mu = mu;
norm_params.fractional.sig = sig;
networks.fractional = train_nn(X_frac_norm, y_frac, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 3. Dense Fractional NN (15 inputs)
fprintf('  Training Dense Fractional NN (15 inputs)... ');
tic;
[X_dense, y_dense] = extract_features(rx_train, bits_train, sym_idx_train, offsets.dense);
[X_dense_norm, mu, sig] = normalize_features(X_dense);
norm_params.dense.mu = mu;
norm_params.dense.sig = sig;
% Use wider first layer for more inputs
networks.dense = train_nn(X_dense_norm, y_dense, [96, 48], max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 4. Structured CNN (7x7)
fprintf('  Training Structured CNN (7x7)... ');
tic;
[X_struct, y_struct] = extract_structured_features(rx_train, bits_train, sym_idx_train, step);
networks.structured = train_cnn(X_struct, y_struct, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

%% ========================================================================
%% TEST OVER SNR RANGE
%% ========================================================================

fprintf('\n[3/3] Testing over SNR range...\n');

% Initialize results
results = struct();
results.threshold.BER = zeros(size(SNR_test));
results.neighbor.BER = zeros(size(SNR_test));
results.fractional.BER = zeros(size(SNR_test));
results.dense.BER = zeros(size(SNR_test));
results.structured.BER = zeros(size(SNR_test));

fprintf('\nSNR(dB) | Threshold | Neighbor | Frac(T/2) | Dense(T/7) | Struct-CNN\n');
fprintf('--------|-----------|----------|-----------|------------|------------\n');

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    rng(100 + snr_idx);
    
    bits_test = randi([0 1], 1, N_test);
    [rx_test, sym_idx_test] = generate_ftn_rx(bits_test, tau, sps, h_srrc, delay, ...
        snr_db, PA_ENABLED, PA_MODEL, pa_params);
    
    margin = max(abs(offsets.dense)) + 10;
    valid_range = (margin+1):(N_test-margin);
    n_valid = length(valid_range);
    
    % Threshold
    bits_th = detect_threshold(rx_test, sym_idx_test, valid_range);
    results.threshold.BER(snr_idx) = mean(bits_th ~= bits_test(valid_range));
    
    % Neighbor NN
    bits_nb = detect_fc(rx_test, sym_idx_test, valid_range, offsets.neighbor, ...
        networks.neighbor, norm_params.neighbor);
    results.neighbor.BER(snr_idx) = mean(bits_nb ~= bits_test(valid_range));
    
    % Fractional NN
    bits_frac = detect_fc(rx_test, sym_idx_test, valid_range, offsets.fractional, ...
        networks.fractional, norm_params.fractional);
    results.fractional.BER(snr_idx) = mean(bits_frac ~= bits_test(valid_range));
    
    % Dense Fractional NN
    bits_dense = detect_fc(rx_test, sym_idx_test, valid_range, offsets.dense, ...
        networks.dense, norm_params.dense);
    results.dense.BER(snr_idx) = mean(bits_dense ~= bits_test(valid_range));
    
    % Structured CNN
    bits_struct = detect_cnn(rx_test, sym_idx_test, valid_range, step, networks.structured);
    results.structured.BER(snr_idx) = mean(bits_struct ~= bits_test(valid_range));
    
    fprintf('  %2d    |  %.2e  |  %.2e | %.2e  | %.2e   | %.2e\n', ...
        snr_db, results.threshold.BER(snr_idx), results.neighbor.BER(snr_idx), ...
        results.fractional.BER(snr_idx), results.dense.BER(snr_idx), ...
        results.structured.BER(snr_idx));
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

figure('Position', [100 100 1200 500]);

% PA Characteristic
subplot(1,2,1);
r_in = linspace(0, 2, 1000);
if PA_ENABLED
    y_out = pa_models(r_in, PA_MODEL, pa_params);
    plot(r_in, abs(y_out), 'b-', 'LineWidth', 2); hold on;
    plot(r_in, r_in, 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel('Input Amplitude'); ylabel('Output Amplitude');
    title(sprintf('PA: %s (IBO=%ddB)', PA_MODEL, IBO_dB));
    legend('PA Output', 'Linear', 'Location', 'best');
else
    text(0.5, 0.5, 'PA Disabled', 'HorizontalAlignment', 'center');
    axis off;
end

% BER Curves
subplot(1,2,2);
colors = lines(5);
semilogy(SNR_test, results.threshold.BER, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
hold on;
semilogy(SNR_test, results.neighbor.BER, 'o-', 'Color', colors(1,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Neighbor NN');
semilogy(SNR_test, results.fractional.BER, 's-', 'Color', colors(2,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Fractional NN (T/2)');
semilogy(SNR_test, results.dense.BER, '^-', 'Color', colors(3,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Dense Frac NN (T/7)');
semilogy(SNR_test, results.structured.BER, 'd-', 'Color', colors(4,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Structured CNN');
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('BER Comparison (\\tau=%.2f, PA=%s)', tau, PA_MODEL));
legend('Location', 'southwest');
ylim([1e-5 1]);

sgtitle('FTN with PA Saturation - NN Detection Comparison', 'FontWeight', 'bold');

% Save
saveas(gcf, 'ftn_pa_nn_results.png');
save('ftn_pa_nn_results.mat', 'results', 'SNR_test', 'tau', 'PA_MODEL', 'IBO_dB');

fprintf('\n========================================\n');
fprintf('Simulation Complete!\n');
fprintf('Figure saved: ftn_pa_nn_results.png\n');
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    
    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);
    
    % Upsample
    tx_up = zeros(N * step, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
    else
        tx_pa = tx_shaped;
    end
    
    % AWGN
    EbN0 = 10^(SNR_dB/10);
    noise_power = 1 / (2 * EbN0);
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    rx = rx_mf(:)' / std(rx_mf);
    
    % Symbol indices (matches reference: delay + 1)
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function [X, y] = extract_features(rx, bits, symbol_indices, offsets)
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples);
    y = zeros(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            X(i, :) = real(rx(indices));
            y(i) = bits(k);
        end
    end
end

function [X_struct, y] = extract_structured_features(rx, bits, symbol_indices, step)
    N = length(bits);
    local_window = -3:3;
    symbol_positions = -3:3;
    
    max_offset = max(symbol_positions) * step + max(local_window);
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_range(i);
        current_center = symbol_indices(k);
        
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            
            if all(indices > 0 & indices <= length(rx))
                X_struct(r, :, 1, i) = real(rx(indices));
            end
        end
        y(i) = bits(k);
    end
    
    % Normalize
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    X_struct = (X_struct - mu) / sig;
end

function [X_norm, mu, sig] = normalize_features(X)
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig == 0) = 1;
    X_norm = (X - mu) ./ sig;
end

function net = train_nn(X, y, hidden_sizes, max_epochs, mini_batch)
    layers = [featureInputLayer(size(X, 2))];
    
    for i = 1:length(hidden_sizes)
        layers = [layers; ...
            fullyConnectedLayer(hidden_sizes(i)); ...
            batchNormalizationLayer; ...
            reluLayer; ...
            dropoutLayer(0.2)];
    end
    
    layers = [layers; ...
        fullyConnectedLayer(2); ...
        softmaxLayer; ...
        classificationLayer];
    
    % Validation split
    n = size(X, 1);
    idx = randperm(n);
    n_val = round(0.1 * n);
    X_val = X(idx(1:n_val), :);
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X(idx(n_val+1:end), :);
    y_tr = categorical(y(idx(n_val+1:end)));
    
    options = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 8, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X_tr, y_tr, layers, options);
end

function net = train_cnn(X_struct, y, max_epochs, mini_batch)
    layers = [
        imageInputLayer([7 7 1], 'Normalization', 'none')
        convolution2dLayer([1 7], 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([7 1], 16, 'Padding', 0)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];
    
    n = size(X_struct, 4);
    idx = randperm(n);
    n_val = round(0.1 * n);
    
    X_val = X_struct(:,:,:,idx(1:n_val));
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X_struct(:,:,:,idx(n_val+1:end));
    y_tr = categorical(y(idx(n_val+1:end)));
    
    options = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 8, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X_tr, y_tr, layers, options);
end

function bits_hat = detect_threshold(rx, symbol_indices, valid_range)
    bits_hat = zeros(1, length(valid_range));
    for i = 1:length(valid_range)
        k = valid_range(i);
        idx = symbol_indices(k);
        if idx > 0 && idx <= length(rx)
            bits_hat(i) = real(rx(idx)) > 0;
        end
    end
end

function bits_hat = detect_fc(rx, symbol_indices, valid_range, offsets, net, norm_params)
    n_valid = length(valid_range);
    n_samples = length(offsets);
    X = zeros(n_valid, n_samples);
    
    for i = 1:n_valid
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            X(i, :) = real(rx(indices));
        end
    end
    
    X_norm = (X - norm_params.mu) ./ norm_params.sig;
    probs = predict(net, X_norm);
    bits_hat = (probs(:, 2) > 0.5)';
end

function bits_hat = detect_cnn(rx, symbol_indices, valid_range, step, net)
    local_window = -3:3;
    symbol_positions = -3:3;
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    
    for i = 1:n_valid
        k = valid_range(i);
        current_center = symbol_indices(k);
        
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            
            if all(indices > 0 & indices <= length(rx))
                X_struct(r, :, 1, i) = real(rx(indices));
            end
        end
    end
    
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    X_struct = (X_struct - mu) / sig;
    
    probs = predict(net, X_struct);
    bits_hat = (probs(:, 2) > 0.5)';
end
