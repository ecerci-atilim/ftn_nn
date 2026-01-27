%% FTN_WITH_PA_SATURATION - Optimized FTN Detection with PA Saturation
%
% Compares multiple detection approaches for FTN signaling with PA saturation:
%   1. Threshold (baseline)
%   2. Neighbor NN (symbol-rate sampling)
%   3. Hybrid NN (optimized sample placement)
%   4. Neighbor+DF (decision feedback)
%   5. Structured CNN (7x7 matrix representation)
%
% Key Optimizations:
%   - Corrected SNR calculation based on actual signal power
%   - Hybrid sampling strategy for better ISI capture
%   - Decision feedback for sequential detection
%   - Proper memory allocation
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

% Simulation Parameters (OPTIMIZED)
SNR_train = 8;              % Training SNR (8dB for better generalization)
SNR_test = 0:2:14;          % Test SNR range
N_train = 100000;           % Training symbols
N_test = 50000;             % Test symbols per SNR

% NN Parameters
hidden_sizes = [64, 32];    % Hidden layer sizes
max_epochs = 50;            % Training epochs
mini_batch = 512;           % Mini-batch size

% Decision Feedback
D_feedback = 4;             % Feedback depth

fprintf('========================================\n');
fprintf('FTN with PA Saturation - OPTIMIZED\n');
fprintf('========================================\n');
fprintf('tau = %.2f, beta = %.2f, sps = %d\n', tau, beta, sps);
fprintf('PA Model: %s, IBO = %d dB, Enabled = %d\n', PA_MODEL, IBO_dB, PA_ENABLED);
fprintf('Training: %d symbols @ %d dB (DF depth=%d)\n', N_train, SNR_train, D_feedback);
fprintf('========================================\n\n');

%% ========================================================================
%% PULSE SHAPING & PA SETUP
%% ========================================================================

% SRRC filter
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
%% COMPUTE OPTIMIZED SAMPLE OFFSETS
%% ========================================================================

% Neighbor: 7 samples at symbol rate
offsets.neighbor = (-3:3) * step;

% Hybrid: optimized placement (center + sub-symbol + neighbors)
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

% Fractional: spread within symbol period
offsets.fractional = round((-3:3) * (step-1) / 3);
offsets.fractional(4) = 0;

fprintf('Sample offsets (step=%d):\n', step);
fprintf('  Neighbor:    [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:      [%s]\n', num2str(offsets.hybrid));
fprintf('  Fractional:  [%s]\n\n', num2str(offsets.fractional));

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/3] Generating training data...\n');
rng(42);
bits_train = randi([0 1], 1, N_train);
[rx_train, sym_idx_train] = generate_ftn_rx_optimized(bits_train, tau, sps, h_srrc, delay, ...
    SNR_train, PA_ENABLED, PA_MODEL, pa_params);

%% ========================================================================
%% TRAIN NEURAL NETWORKS
%% ========================================================================

fprintf('\n[2/3] Training neural networks...\n');

networks = struct();
norm_params = struct();

% 1. Neighbor NN (7 inputs)
fprintf('  Training Neighbor NN... ');
tic;
[X_nb, y_nb] = extract_features(rx_train, bits_train, sym_idx_train, offsets.neighbor);
[X_nb_norm, mu, sig] = normalize_features(X_nb);
norm_params.neighbor.mu = mu;
norm_params.neighbor.sig = sig;
networks.neighbor = train_nn(X_nb_norm, y_nb, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 2. Hybrid NN (7 inputs, optimized placement)
fprintf('  Training Hybrid NN... ');
tic;
[X_hyb, y_hyb] = extract_features(rx_train, bits_train, sym_idx_train, offsets.hybrid);
[X_hyb_norm, mu, sig] = normalize_features(X_hyb);
norm_params.hybrid.mu = mu;
norm_params.hybrid.sig = sig;
networks.hybrid = train_nn(X_hyb_norm, y_hyb, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 3. Neighbor with Decision Feedback (11 inputs = 7 + 4 DF)
fprintf('  Training Neighbor+DF NN (D=%d)... ', D_feedback);
tic;
[X_nb_df, y_nb_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, ...
    offsets.neighbor, D_feedback);
[X_nb_df_norm, mu, sig] = normalize_features(X_nb_df);
norm_params.neighbor_df.mu = mu;
norm_params.neighbor_df.sig = sig;
norm_params.neighbor_df.D = D_feedback;
networks.neighbor_df = train_nn(X_nb_df_norm, y_nb_df, [80, 40], max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 4. Hybrid with Decision Feedback (11 inputs)
fprintf('  Training Hybrid+DF NN (D=%d)... ', D_feedback);
tic;
[X_hyb_df, y_hyb_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, ...
    offsets.hybrid, D_feedback);
[X_hyb_df_norm, mu, sig] = normalize_features(X_hyb_df);
norm_params.hybrid_df.mu = mu;
norm_params.hybrid_df.sig = sig;
norm_params.hybrid_df.D = D_feedback;
networks.hybrid_df = train_nn(X_hyb_df_norm, y_hyb_df, [80, 40], max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 5. Structured CNN (7x7)
fprintf('  Training Structured CNN (7x7)... ');
tic;
[X_struct, y_struct, mu_cnn, sig_cnn] = extract_structured_features_normalized(rx_train, bits_train, sym_idx_train, step);
norm_params.structured.mu = mu_cnn;
norm_params.structured.sig = sig_cnn;
networks.structured = train_cnn(X_struct, y_struct, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

%% ========================================================================
%% TEST OVER SNR RANGE
%% ========================================================================

fprintf('\n[3/3] Testing over SNR range...\n');

% Initialize results
results = struct();
methods = {'threshold', 'neighbor', 'hybrid', 'neighbor_df', 'hybrid_df', 'structured'};
for m = 1:length(methods)
    results.(methods{m}).BER = zeros(size(SNR_test));
end

fprintf('\nSNR(dB) | Threshold | Neighbor | Hybrid | Neigh+DF | Hyb+DF | Struct\n');
fprintf('--------|-----------|----------|--------|----------|--------|--------\n');

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    rng(100 + snr_idx);
    
    bits_test = randi([0 1], 1, N_test);
    [rx_test, sym_idx_test] = generate_ftn_rx_optimized(bits_test, tau, sps, h_srrc, delay, ...
        snr_db, PA_ENABLED, PA_MODEL, pa_params);
    
    margin = 3*step + D_feedback + 20;
    valid_range = (margin+1):(N_test-margin);
    
    % Threshold
    bits_th = detect_threshold(rx_test, sym_idx_test, valid_range);
    results.threshold.BER(snr_idx) = mean(bits_th ~= bits_test(valid_range));
    
    % Neighbor NN
    bits_nb = detect_fc(rx_test, sym_idx_test, valid_range, offsets.neighbor, ...
        networks.neighbor, norm_params.neighbor);
    results.neighbor.BER(snr_idx) = mean(bits_nb ~= bits_test(valid_range));
    
    % Hybrid NN
    bits_hyb = detect_fc(rx_test, sym_idx_test, valid_range, offsets.hybrid, ...
        networks.hybrid, norm_params.hybrid);
    results.hybrid.BER(snr_idx) = mean(bits_hyb ~= bits_test(valid_range));
    
    % Neighbor+DF (sequential detection)
    bits_nb_df = detect_fc_df(rx_test, bits_test, sym_idx_test, valid_range, ...
        offsets.neighbor, networks.neighbor_df, norm_params.neighbor_df);
    results.neighbor_df.BER(snr_idx) = mean(bits_nb_df ~= bits_test(valid_range));
    
    % Hybrid+DF (sequential detection)
    bits_hyb_df = detect_fc_df(rx_test, bits_test, sym_idx_test, valid_range, ...
        offsets.hybrid, networks.hybrid_df, norm_params.hybrid_df);
    results.hybrid_df.BER(snr_idx) = mean(bits_hyb_df ~= bits_test(valid_range));
    
    % Structured CNN
    bits_struct = detect_cnn(rx_test, sym_idx_test, valid_range, step, ...
        networks.structured, norm_params.structured);
    results.structured.BER(snr_idx) = mean(bits_struct ~= bits_test(valid_range));
    
    fprintf('  %2d    | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e\n', ...
        snr_db, results.threshold.BER(snr_idx), results.neighbor.BER(snr_idx), ...
        results.hybrid.BER(snr_idx), results.neighbor_df.BER(snr_idx), ...
        results.hybrid_df.BER(snr_idx), results.structured.BER(snr_idx));
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

figure('Position', [100 100 1400 500]);

% PA Characteristic
subplot(1,3,1);
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
subplot(1,3,[2 3]);
colors = lines(6);
semilogy(SNR_test, results.threshold.BER, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
hold on;
semilogy(SNR_test, results.neighbor.BER, 'o-', 'Color', colors(1,:), ...
    'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Neighbor');
semilogy(SNR_test, results.hybrid.BER, 's-', 'Color', colors(2,:), ...
    'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Hybrid');
semilogy(SNR_test, results.neighbor_df.BER, '^-', 'Color', colors(3,:), ...
    'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'Neighbor+DF');
semilogy(SNR_test, results.hybrid_df.BER, 'v-', 'Color', colors(4,:), ...
    'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'Hybrid+DF');
semilogy(SNR_test, results.structured.BER, 'd-', 'Color', colors(5,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Struct-CNN');
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('BER Comparison (\\tau=%.2f, PA=%s, D=%d)', tau, PA_MODEL, D_feedback));
legend('Location', 'southwest');
ylim([1e-5 1]);

sgtitle('FTN with PA Saturation - Optimized NN Detection', 'FontWeight', 'bold');

% Save
saveas(gcf, 'ftn_pa_nn_optimized.png');
save('ftn_pa_nn_optimized.mat', 'results', 'SNR_test', 'tau', 'PA_MODEL', 'IBO_dB', ...
    'networks', 'norm_params', 'offsets');

fprintf('\n========================================\n');
fprintf('Simulation Complete!\n');
fprintf('Best @10dB: Hybrid+DF = %.2e\n', results.hybrid_df.BER(SNR_test==10));
fprintf('Figure saved: ftn_pa_nn_optimized.png\n');
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_optimized(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % OPTIMIZED signal generation
    
    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);
    
    % Exact-size upsampling
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation (force real for BPSK)
    if pa_enabled
        tx_pa = real(pa_models(tx_shaped, pa_model, pa_params));
    else
        tx_pa = tx_shaped;
    end
    
    % Corrected SNR: based on actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter and normalize
    rx_mf = conv(rx_noisy, h, 'full');
    rx = rx_mf(:)' / std(rx_mf);
    
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
    actual = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices), continue; end
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            actual = actual + 1;
            X(actual, :) = real(rx(indices));
            y(actual) = bits(k);
        end
    end
    X = X(1:actual, :);
    y = y(1:actual);
end

function [X, y] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Features + decision feedback (teacher forcing during training)
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + D + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples + D);
    y = zeros(n_valid, 1);
    actual = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices) || k <= D, continue; end
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            actual = actual + 1;
            X(actual, 1:n_samples) = real(rx(indices));
            X(actual, n_samples+1:end) = 2*bits(k-D:k-1) - 1;  % Teacher forcing
            y(actual) = bits(k);
        end
    end
    X = X(1:actual, :);
    y = y(1:actual);
end

function [X_struct, y, mu, sig] = extract_structured_features_normalized(rx, bits, symbol_indices, step)
    N = length(bits);
    local_window = -3:3;
    symbol_positions = -3:3;
    
    max_offset = max(abs(symbol_positions)) * step + max(abs(local_window));
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);
    actual = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices), continue; end
        current_center = symbol_indices(k);
        
        valid = true;
        temp = zeros(7, 7);
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            
            if all(indices > 0 & indices <= length(rx))
                temp(r, :) = real(rx(indices));
            else
                valid = false;
                break;
            end
        end
        
        if valid
            actual = actual + 1;
            X_struct(:, :, 1, actual) = temp;
            y(actual) = bits(k);
        end
    end
    
    X_struct = X_struct(:, :, :, 1:actual);
    y = y(1:actual);
    
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    if sig == 0, sig = 1; end
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
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
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
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X_tr, y_tr, layers, options);
end

function bits_hat = detect_threshold(rx, symbol_indices, valid_range)
    bits_hat = zeros(1, length(valid_range));
    for i = 1:length(valid_range)
        k = valid_range(i);
        if k <= length(symbol_indices)
            idx = symbol_indices(k);
            if idx > 0 && idx <= length(rx)
                bits_hat(i) = real(rx(idx)) > 0;
            end
        end
    end
end

function bits_hat = detect_fc(rx, symbol_indices, valid_range, offsets, net, norm_params)
    n_valid = length(valid_range);
    n_samples = length(offsets);
    X = zeros(n_valid, n_samples);
    
    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices)
            center = symbol_indices(k);
            indices = center + offsets;
            if all(indices > 0 & indices <= length(rx))
                X(i, :) = real(rx(indices));
            end
        end
    end
    
    X_norm = (X - norm_params.mu) ./ norm_params.sig;
    probs = predict(net, X_norm);
    bits_hat = (probs(:, 2) > 0.5)';
end

function bits_hat = detect_fc_df(rx, bits_init, symbol_indices, valid_range, offsets, net, norm_params)
    % Sequential detection with decision feedback
    D = norm_params.D;
    N = length(symbol_indices);
    bits_full = zeros(1, N);
    
    % Initialize edges with true bits
    margin = valid_range(1) - 1;
    bits_full(1:margin) = bits_init(1:margin);
    
    n_samples = length(offsets);
    
    for i = 1:length(valid_range)
        k = valid_range(i);
        if k > length(symbol_indices) || k <= D
            continue;
        end
        
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            x_samples = real(rx(indices));
            fb = 2*bits_full(k-D:k-1) - 1;
            x = [x_samples, fb];
            x_norm = (x - norm_params.mu) ./ norm_params.sig;
            
            prob = predict(net, x_norm);
            bits_full(k) = (prob(2) > 0.5);
        end
    end
    
    bits_hat = bits_full(valid_range);
end

function bits_hat = detect_cnn(rx, symbol_indices, valid_range, step, net, norm_params)
    local_window = -3:3;
    symbol_positions = -3:3;
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices), continue; end
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
    
    % Use training normalization
    X_struct = (X_struct - norm_params.mu) / norm_params.sig;
    
    probs = predict(net, X_struct);
    bits_hat = (probs(:, 2) > 0.5)';
end
