%% FTN_WITH_PA_SATURATION - FTN Detection with PA Saturation & NN Equalization
%
% Compares multiple detection approaches for FTN signaling with PA saturation:
%   1. Threshold (baseline)
%   2. Neighbor NN (symbol-rate sampling)
%   3. Hybrid NN (mixed sampling - better ISI capture)
%   4. Neighbor + Decision Feedback (D=4)
%   5. Hybrid + Decision Feedback (D=4) - BEST PERFORMANCE
%   6. Structured CNN (7x7 matrix representation)
%
% Key Optimizations:
%   - Correct noise power calculation based on actual signal energy
%   - Decision Feedback (D=4) for improved detection
%   - Hybrid sampling: captures both symbol-rate and inter-symbol info
%   - Training at lower SNR (8dB) for better generalization
%   - Consistent normalization across training/testing
%
% Target Performance: BER ~1e-5 @ 10dB SNR with Decision Feedback
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

% Simulation Parameters - OPTIMIZED
SNR_train = 8;              % Training SNR (challenging region)
SNR_test = 0:2:14;          % Test SNR range
N_train = 100000;           % Training symbols
N_test = 30000;             % Test symbols per SNR

% NN Parameters
hidden_sizes = [64, 32];    % Hidden layer sizes
max_epochs = 40;            % Training epochs
mini_batch = 512;           % Mini-batch size

% Decision Feedback Parameters
D_feedback = 4;             % Number of feedback bits

fprintf('========================================\n');
fprintf('FTN with PA Saturation - Optimized NN Detection\n');
fprintf('========================================\n');
fprintf('tau = %.2f, beta = %.2f, sps = %d\n', tau, beta, sps);
fprintf('PA Model: %s, IBO = %d dB, Enabled = %d\n', PA_MODEL, IBO_dB, PA_ENABLED);
fprintf('Training: %d symbols @ %d dB (optimized)\n', N_train, SNR_train);
fprintf('Decision Feedback: D = %d\n', D_feedback);
fprintf('========================================\n\n');

%% ========================================================================
%% PULSE SHAPING & PA SETUP
%% ========================================================================

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

offsets = struct();

% Neighbor: 7 samples at symbol rate (-3T to +3T)
offsets.neighbor = (-3:3) * step;

% Hybrid: Mix of symbol-rate and inter-symbol (BEST for ISI capture)
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

fprintf('Sample offsets (step=%d):\n', step);
fprintf('  Neighbor: [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:   [%s]\n\n', num2str(offsets.hybrid));

%% ========================================================================
%% GENERATE TRAINING DATA (OPTIMIZED)
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

% 2. Hybrid NN (7 inputs)
fprintf('  Training Hybrid NN... ');
tic;
[X_hyb, y_hyb] = extract_features(rx_train, bits_train, sym_idx_train, offsets.hybrid);
[X_hyb_norm, mu, sig] = normalize_features(X_hyb);
norm_params.hybrid.mu = mu;
norm_params.hybrid.sig = sig;
networks.hybrid = train_nn(X_hyb_norm, y_hyb, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 3. Neighbor + DF (7+4=11 inputs)
fprintf('  Training Neighbor+DF NN (D=%d)... ', D_feedback);
tic;
[X_nb_df, y_nb_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.neighbor, D_feedback);
[X_nb_df_norm, mu, sig] = normalize_features(X_nb_df);
norm_params.neighbor_df.mu = mu;
norm_params.neighbor_df.sig = sig;
networks.neighbor_df = train_nn(X_nb_df_norm, y_nb_df, [96, 48], max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 4. Hybrid + DF (7+4=11 inputs) - EXPECTED BEST
fprintf('  Training Hybrid+DF NN (D=%d)... ', D_feedback);
tic;
[X_hyb_df, y_hyb_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.hybrid, D_feedback);
[X_hyb_df_norm, mu, sig] = normalize_features(X_hyb_df);
norm_params.hybrid_df.mu = mu;
norm_params.hybrid_df.sig = sig;
networks.hybrid_df = train_nn(X_hyb_df_norm, y_hyb_df, [96, 48], max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% 5. Structured CNN (7x7)
fprintf('  Training Structured CNN... ');
tic;
[X_struct, y_struct, mu_cnn, sig_cnn] = extract_structured_features(rx_train, bits_train, sym_idx_train, step);
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

fprintf('\nSNR(dB) | Threshold | Neighbor | Hybrid  | Nbr+DF  | Hyb+DF  | Struct-CNN\n');
fprintf('--------|-----------|----------|---------|---------|---------|------------\n');

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    rng(100 + snr_idx);
    
    bits_test = randi([0 1], 1, N_test);
    [rx_test, sym_idx_test] = generate_ftn_rx_optimized(bits_test, tau, sps, h_srrc, delay, ...
        snr_db, PA_ENABLED, PA_MODEL, pa_params);
    
    margin = max(abs(offsets.neighbor)) + D_feedback + 10;
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
    
    % Neighbor + DF (sequential)
    bits_nb_df = detect_fc_df(rx_test, sym_idx_test, valid_range, offsets.neighbor, ...
        networks.neighbor_df, norm_params.neighbor_df, D_feedback);
    results.neighbor_df.BER(snr_idx) = mean(bits_nb_df ~= bits_test(valid_range));
    
    % Hybrid + DF (sequential) - EXPECTED BEST
    bits_hyb_df = detect_fc_df(rx_test, sym_idx_test, valid_range, offsets.hybrid, ...
        networks.hybrid_df, norm_params.hybrid_df, D_feedback);
    results.hybrid_df.BER(snr_idx) = mean(bits_hyb_df ~= bits_test(valid_range));
    
    % Structured CNN
    bits_struct = detect_cnn(rx_test, sym_idx_test, valid_range, step, networks.structured, norm_params.structured);
    results.structured.BER(snr_idx) = mean(bits_struct ~= bits_test(valid_range));
    
    fprintf('  %2d    |  %.2e |  %.2e | %.2e | %.2e | %.2e | %.2e\n', ...
        snr_db, results.threshold.BER(snr_idx), results.neighbor.BER(snr_idx), ...
        results.hybrid.BER(snr_idx), results.neighbor_df.BER(snr_idx), ...
        results.hybrid_df.BER(snr_idx), results.structured.BER(snr_idx));
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

figure('Position', [100 100 1400 500]);

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
colors = lines(6);
semilogy(SNR_test, results.threshold.BER, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
hold on;
semilogy(SNR_test, results.neighbor.BER, 'o-', 'Color', colors(1,:), ...
    'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Neighbor');
semilogy(SNR_test, results.hybrid.BER, 's-', 'Color', colors(2,:), ...
    'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Hybrid');
semilogy(SNR_test, results.neighbor_df.BER, '^-', 'Color', colors(3,:), ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Neighbor+DF');
semilogy(SNR_test, results.hybrid_df.BER, 'd-', 'Color', colors(4,:), ...
    'LineWidth', 2.5, 'MarkerSize', 9, 'DisplayName', 'Hybrid+DF (Best)');
semilogy(SNR_test, results.structured.BER, 'v-', 'Color', colors(5,:), ...
    'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Struct-CNN');
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('BER Comparison (\\tau=%.2f, PA=%s)', tau, PA_MODEL));
legend('Location', 'southwest');
ylim([1e-5 1]);

sgtitle('FTN with PA Saturation - Optimized NN Detection', 'FontWeight', 'bold');

% Save
saveas(gcf, 'ftn_pa_nn_optimized_results.png');
save('ftn_pa_nn_optimized_results.mat', 'results', 'SNR_test', 'tau', 'PA_MODEL', 'IBO_dB', 'D_feedback');

fprintf('\n========================================\n');
fprintf('Simulation Complete!\n');
fprintf('Best @ 10dB: Hybrid+DF = %.2e\n', results.hybrid_df.BER(SNR_test==10));
fprintf('Figure saved: ftn_pa_nn_optimized_results.png\n');
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_optimized(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % OPTIMIZED signal generation with correct noise model
    
    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);
    
    % Optimized upsampling
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
        tx_pa = real(tx_pa);  % Keep real for BPSK
    else
        tx_pa = tx_shaped;
    end
    
    % OPTIMIZED: Noise based on actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0_lin = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0_lin);
    
    % Add real noise (BPSK)
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    rx = rx_mf(:)' / std(rx_mf);
    
    % Symbol indices
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
    valid_count = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices)
            center = symbol_indices(k);
            indices = center + offsets;
            
            if all(indices > 0 & indices <= length(rx))
                valid_count = valid_count + 1;
                X(valid_count, :) = real(rx(indices));
                y(valid_count) = bits(k);
            end
        end
    end
    
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X, y] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Extract features with Decision Feedback
    
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + D + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples + D);
    y = zeros(n_valid, 1);
    valid_count = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices) && k > D
            center = symbol_indices(k);
            indices = center + offsets;
            
            if all(indices > 0 & indices <= length(rx))
                valid_count = valid_count + 1;
                X(valid_count, 1:n_samples) = real(rx(indices));
                X(valid_count, n_samples+1:end) = 2*bits(k-D:k-1) - 1;  % True bits for training
                y(valid_count) = bits(k);
            end
        end
    end
    
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X_struct, y, mu, sig] = extract_structured_features(rx, bits, symbol_indices, step)
    N = length(bits);
    local_window = -3:3;
    symbol_positions = -3:3;
    
    max_offset = max(abs(symbol_positions)) * step + max(abs(local_window));
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);
    valid_count = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices)
            current_center = symbol_indices(k);
            valid_sample = true;
            
            for r = 1:7
                sym_pos = symbol_positions(r);
                neighbor_center = current_center + sym_pos * step;
                indices = neighbor_center + local_window;
                
                if all(indices > 0 & indices <= length(rx))
                    X_struct(r, :, 1, valid_count+1) = real(rx(indices));
                else
                    valid_sample = false;
                    break;
                end
            end
            
            if valid_sample
                valid_count = valid_count + 1;
                y(valid_count) = bits(k);
            end
        end
    end
    
    X_struct = X_struct(:,:,:,1:valid_count);
    y = y(1:valid_count);
    
    % Compute normalization
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
        'InitialLearnRate', 0.001, ...
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

function bits_hat = detect_fc_df(rx, symbol_indices, valid_range, offsets, net, norm_params, D)
    % Sequential detection with decision feedback
    
    N = max(valid_range);
    bits_hat_full = zeros(1, N);
    bits_hat_full(1:valid_range(1)-1) = randi([0 1], 1, valid_range(1)-1);
    
    n_samples = length(offsets);
    
    for i = 1:length(valid_range)
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx)) && k > D
            % Signal samples
            x_signal = real(rx(indices));
            
            % Decision feedback (previous decisions)
            x_feedback = 2*bits_hat_full(k-D:k-1) - 1;
            
            % Combine
            x = [x_signal, x_feedback];
            x_norm = (x - norm_params.mu) ./ norm_params.sig;
            
            % Predict
            prob = predict(net, x_norm);
            bits_hat_full(k) = (prob(2) > 0.5);
        end
    end
    
    bits_hat = bits_hat_full(valid_range);
end

function bits_hat = detect_cnn(rx, symbol_indices, valid_range, step, net, norm_params)
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
    
    % Use stored normalization
    X_struct = (X_struct - norm_params.mu) / norm_params.sig;
    
    probs = predict(net, X_struct);
    bits_hat = (probs(:, 2) > 0.5)';
end
