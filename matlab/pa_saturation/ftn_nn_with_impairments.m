%% FTN Neural Network Detection with Hardware Impairments
%
% This simulation combines:
%   1. Power Amplifier (PA) saturation and hardware impairments
%   2. Window-based neural network detection approaches
%   3. Comparison of detection methods (Neighbor, Fractional, Hybrid, Structured CNN)
%
% Key Innovation:
%   - Window-based detectors capture nonlinear distortion better than symbol-rate sampling
%   - Neural networks learn optimal detection in presence of PA saturation
%   - Structured CNN processes ISI more effectively than FC networks
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% FTN Parameters
tau = 0.7;                  % FTN compression factor
beta = 0.3;                 % SRRC roll-off
sps = 10;                   % Samples per symbol
span = 6;                   % Pulse span in symbols

% Hardware Impairment Configuration
config.pa_enabled = true;
config.pa_model = 'rapp';           % 'rapp', 'saleh', or 'soft_limiter'
config.pa_ibo_db = 3;               % Input Back-Off (dB)

config.iq_tx_enabled = false;       % TX IQ imbalance
config.iq_tx_amp = 0.1;
config.iq_tx_phase = 5;             % degrees

config.phase_noise_enabled = false; % Phase noise
config.pn_variance = 0.01;

config.cfo_enabled = false;         % Carrier frequency offset
config.cfo_hz = 100;
config.sample_rate = sps * 1000;

% Simulation Parameters
SNR_train = 10;             % Training SNR (dB)
SNR_test = 0:2:14;          % Test SNR range (dB)
N_train = 50000;            % Training symbols
N_test = 20000;             % Test symbols per SNR point
min_errors = 100;           % Minimum errors for BER calculation

% Neural Network Parameters
hidden_sizes = [32, 16];    % Hidden layer sizes
max_epochs = 30;            % Training epochs
mini_batch = 512;           % Mini-batch size
use_decision_feedback = false;  % Decision feedback (future enhancement)
D_feedback = 4;             % Feedback depth

fprintf('========================================\n');
fprintf('FTN NN Detection with Impairments\n');
fprintf('========================================\n');
fprintf('tau = %.2f, beta = %.2f, sps = %d\n', tau, beta, sps);
fprintf('PA: %s (IBO=%ddB, enabled=%d)\n', config.pa_model, config.pa_ibo_db, config.pa_enabled);
fprintf('Training: %d symbols @ SNR=%ddB\n', N_train, SNR_train);
fprintf('Testing: SNR %d:%d:%d dB\n', SNR_test(1), SNR_test(2)-SNR_test(1), SNR_test(end));
fprintf('========================================\n\n');

%% ========================================================================
%% PULSE SHAPING FILTER
%% ========================================================================

h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;
step = round(tau * sps);

fprintf('Generated SRRC pulse: span=%d symbols, delay=%d samples, step=%d samples\n', ...
        span, delay, step);

%% ========================================================================
%% COMPUTE SAMPLE OFFSETS FOR EACH DETECTION APPROACH
%% ========================================================================

% Approach 1: Neighbor (symbol-rate sampling)
% Sample at 7 neighboring symbol instants: -3T, -2T, -1T, 0, +T, +2T, +3T
offsets.neighbor = (-3:3) * step;

% Approach 2: Fractional (T/2-spaced equalizer)
% Classical fractional spacing: samples at half-symbol intervals
% -3T/2, -2T/2, -T/2, 0, T/2, 2T/2, 3T/2
frac_step = round(step / 2);  % T/2 spacing
offsets.fractional = (-3:3) * frac_step;

% Approach 3: Hybrid (T/3-spaced for denser sampling)
% Samples at T/3 intervals for even finer resolution
hybrid_step = round(step / 3);
offsets.hybrid = (-3:3) * hybrid_step;

fprintf('Sample offsets computed:\n');
fprintf('  Neighbor:    [%s]\n', sprintf('%d ', offsets.neighbor));
fprintf('  Fractional:  [%s]\n', sprintf('%d ', offsets.fractional));
fprintf('  Hybrid:      [%s]\n\n', sprintf('%d ', offsets.hybrid));

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/3] Generating training data...\n');
rng(42);  % Reproducibility
bits_train = randi([0 1], 1, N_train);
[rx_train, symbol_indices] = generate_ftn_rx_with_impairments(...
    bits_train, tau, sps, h, delay, SNR_train, config);

fprintf('  Generated %d symbols, rx length=%d samples\n', N_train, length(rx_train));

%% ========================================================================
%% TRAIN NEURAL NETWORKS FOR EACH APPROACH
%% ========================================================================

fprintf('\n[2/3] Training neural networks...\n');

% Store networks
networks = struct();

% Approach 1: Neighbor
fprintf('  [1/4] Training Neighbor detector... ');
tic;
[X_train, y_train] = extract_features(rx_train, bits_train, symbol_indices, ...
                                       offsets.neighbor);
networks.neighbor = train_nn(X_train, y_train, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% Approach 2: Fractional
fprintf('  [2/4] Training Fractional detector... ');
tic;
[X_train, y_train] = extract_features(rx_train, bits_train, symbol_indices, ...
                                       offsets.fractional);
networks.fractional = train_nn(X_train, y_train, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% Approach 3: Hybrid
fprintf('  [3/4] Training Hybrid detector... ');
tic;
[X_train, y_train] = extract_features(rx_train, bits_train, symbol_indices, ...
                                       offsets.hybrid);
networks.hybrid = train_nn(X_train, y_train, hidden_sizes, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

% Approach 4: Structured CNN
fprintf('  [4/4] Training Structured CNN... ');
tic;
[X_train_struct, y_train_struct] = extract_structured_features(rx_train, bits_train, ...
                                                                symbol_indices, step);
networks.structured = train_cnn(X_train_struct, y_train_struct, max_epochs, mini_batch);
fprintf('done (%.1fs)\n', toc);

%% ========================================================================
%% TEST OVER SNR RANGE
%% ========================================================================

fprintf('\n[3/3] Testing performance over SNR range...\n');

% Initialize results
approaches = {'neighbor', 'fractional', 'hybrid', 'structured'};
results = struct();
for i = 1:length(approaches)
    results.(approaches{i}).BER = zeros(size(SNR_test));
    results.(approaches{i}).SNR = SNR_test;
end

% Test each SNR point
for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    fprintf('  SNR = %2d dB: ', snr_db);

    % Generate test data
    rng(100 + snr_idx);
    bits_test = randi([0 1], 1, N_test);
    [rx_test, sym_idx_test] = generate_ftn_rx_with_impairments(...
        bits_test, tau, sps, h, delay, snr_db, config);

    % Test each approach
    for app_idx = 1:length(approaches)
        app_name = approaches{app_idx};

        if strcmp(app_name, 'structured')
            % Structured CNN uses different feature extraction
            [X_test, valid_bits] = extract_structured_features(rx_test, bits_test, ...
                                                                sym_idx_test, step);
            bits_hat = detect_cnn(X_test, networks.structured);
        else
            % FC networks use simple feature vectors
            off = offsets.(app_name);
            [X_test, valid_bits] = extract_features(rx_test, bits_test, ...
                                                     sym_idx_test, off);
            bits_hat = detect_fc(X_test, networks.(app_name));
        end

        % Calculate BER
        errors = sum(bits_hat ~= valid_bits);
        ber = errors / length(valid_bits);
        results.(app_name).BER(snr_idx) = ber;
    end

    fprintf('Neighbor=%.2e, Frac=%.2e, Hybrid=%.2e, Struct=%.2e\n', ...
            results.neighbor.BER(snr_idx), results.fractional.BER(snr_idx), ...
            results.hybrid.BER(snr_idx), results.structured.BER(snr_idx));
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

fprintf('\nGenerating plots...\n');
figure('Position', [100 100 1400 900]);

% Subplot 1: PA Characteristic (if enabled)
subplot(2,3,1);
if config.pa_enabled
    r_in = linspace(0, 2, 1000);
    pa_params = get_pa_params(config);
    y_out = pa_models(r_in, config.pa_model, pa_params);
    plot(r_in, abs(y_out), 'b-', 'LineWidth', 2); hold on;
    plot(r_in, r_in, 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel('Input Amplitude');
    ylabel('Output Amplitude');
    title(sprintf('PA: %s (IBO=%ddB)', config.pa_model, config.pa_ibo_db));
    legend('PA Output', 'Linear', 'Location', 'best');
else
    text(0.5, 0.5, 'PA Disabled', 'HorizontalAlignment', 'center', 'FontSize', 14);
    axis off;
end

% Subplot 2: Sample Offsets Visualization
subplot(2,3,2);
hold on;
plot(offsets.neighbor, ones(size(offsets.neighbor)), 'ro', 'MarkerSize', 10, ...
     'LineWidth', 2, 'DisplayName', 'Neighbor');
plot(offsets.fractional, 2*ones(size(offsets.fractional)), 'bs', 'MarkerSize', 10, ...
     'LineWidth', 2, 'DisplayName', 'Fractional');
plot(offsets.hybrid, 3*ones(size(offsets.hybrid)), 'g^', 'MarkerSize', 10, ...
     'LineWidth', 2, 'DisplayName', 'Hybrid');
ylim([0 4]);
grid on;
xlabel('Sample Offset from Symbol Center');
ylabel('Approach');
title('Sampling Strategies');
legend('Location', 'best');
set(gca, 'YTick', [1 2 3], 'YTickLabel', {'Neighbor', 'Fractional', 'Hybrid'});

% Subplot 3: BER Comparison
subplot(2,3,[3,6]);
colors = lines(4);
semilogy(SNR_test, results.neighbor.BER, 'o-', 'Color', colors(1,:), ...
         'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Neighbor');
hold on;
semilogy(SNR_test, results.fractional.BER, 's-', 'Color', colors(2,:), ...
         'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Fractional');
semilogy(SNR_test, results.hybrid.BER, '^-', 'Color', colors(3,:), ...
         'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hybrid');
semilogy(SNR_test, results.structured.BER, 'd-', 'Color', colors(4,:), ...
         'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Structured CNN');
grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Bit Error Rate', 'FontSize', 12);
title(sprintf('BER Performance (\\tau=%.2f, PA=%s)', tau, config.pa_model), 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-4 1]);

% Subplot 4: Training Data Constellation
subplot(2,3,4);
[rx_demo, ~] = generate_ftn_rx_with_impairments(...
    randi([0 1], 1, 1000), tau, sps, h, delay, 10, config);
plot(real(rx_demo), imag(rx_demo), 'b.', 'MarkerSize', 3);
grid on; axis equal;
xlabel('In-Phase'); ylabel('Quadrature');
title('Received Signal Constellation (After Impairments)');

% Subplot 5: Performance Summary Table
subplot(2,3,5);
axis off;
% Create summary text
summary_text = {
    sprintf('\\bfConfiguration:'), ...
    sprintf('  \\tau = %.2f', tau), ...
    sprintf('  PA: %s (IBO=%ddB)', config.pa_model, config.pa_ibo_db), ...
    sprintf('  Training: %d symbols', N_train), ...
    '', ...
    sprintf('\\bfBest BER at %ddB:', SNR_test(end)), ...
    sprintf('  Neighbor:     %.2e', results.neighbor.BER(end)), ...
    sprintf('  Fractional:   %.2e', results.fractional.BER(end)), ...
    sprintf('  Hybrid:       %.2e', results.hybrid.BER(end)), ...
    sprintf('  Structured:   %.2e', results.structured.BER(end))
};
text(0.1, 0.9, summary_text, 'VerticalAlignment', 'top', 'FontSize', 10, 'Interpreter', 'tex');

sgtitle('FTN Neural Network Detection with Hardware Impairments', 'FontSize', 15, 'FontWeight', 'bold');

% Save results
saveas(gcf, 'matlab/pa_saturation/ftn_nn_impairments_results.png');
save('matlab/pa_saturation/ftn_nn_results.mat', 'results', 'config', 'SNR_test', 'tau');

fprintf('\nSimulation complete!\n');
fprintf('Results saved to: matlab/pa_saturation/ftn_nn_impairments_results.png\n');
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_with_impairments(bits, tau, sps, h, delay, SNR_dB, config)
    % Generate FTN received signal with hardware impairments

    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);

    % Upsample
    tx_up = zeros(N * step, 1);
    tx_up(1:step:end) = symbols;

    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');

    % Apply TX impairments
    tx_impaired = apply_tx_impairments(tx_shaped, config);

    % AWGN Channel
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    noise = sqrt(noise_var) * (randn(size(tx_impaired)) + 1j*randn(size(tx_impaired)));
    rx_noisy = tx_impaired + noise;

    % Apply RX impairments
    rx_impaired = apply_rx_impairments(rx_noisy, config);

    % Matched filter
    rx_mf = conv(rx_impaired, h, 'full');

    % Normalize
    rx = rx_mf(:)' / std(rx_mf);

    % Symbol indices (accounting for full convolution and MATLAB indexing)
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function tx_out = apply_tx_impairments(tx_in, config)
    % Apply transmitter impairments
    tx_out = tx_in;

    % IQ Imbalance (TX)
    if config.iq_tx_enabled
        alpha = config.iq_tx_amp;
        phi = config.iq_tx_phase * pi / 180;
        I = real(tx_out);
        Q = imag(tx_out);
        tx_out = (1+alpha)*I + 1j*(1-alpha)*Q.*exp(1j*phi);
    end

    % PA Saturation
    if config.pa_enabled
        pa_params = get_pa_params(config);
        tx_out = pa_models(tx_out, config.pa_model, pa_params);
    end
end

function rx_out = apply_rx_impairments(rx_in, config)
    % Apply receiver impairments
    rx_out = rx_in;

    % Carrier Frequency Offset
    if config.cfo_enabled
        n = (0:length(rx_out)-1)';
        cfo_phase = 2 * pi * config.cfo_hz * n / config.sample_rate;
        rx_out = rx_out .* exp(1j * cfo_phase);
    end

    % Phase Noise
    if config.phase_noise_enabled
        pn = cumsum(sqrt(config.pn_variance) * randn(size(rx_out)));
        rx_out = rx_out .* exp(1j * pn);
    end
end

function pa_params = get_pa_params(config)
    % Get PA parameters based on configuration
    IBO_lin = 10^(config.pa_ibo_db/10);

    switch lower(config.pa_model)
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
end

function [X, y] = extract_features(rx, bits, symbol_indices, offsets)
    % Extract feature vectors for FC network
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + 10;  % Safety margin

    % Valid range (exclude edges)
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);

    % Preallocate
    X = zeros(n_valid, n_samples);
    y = zeros(n_valid, 1);

    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices)
            center = symbol_indices(k);
            indices = center + offsets;

            % Bounds checking
            valid_idx = (indices > 0) & (indices <= length(rx));
            if all(valid_idx)
                X(i, :) = real(rx(indices));  % Take real part for BPSK
                y(i) = bits(k);
            end
        end
    end
end

function [X_struct, y] = extract_structured_features(rx, bits, symbol_indices, step)
    % Extract structured 7x7 matrices for CNN
    % Each matrix: 7 neighbor symbols (-3 to +3) × 7 samples around each

    N = length(bits);
    local_window = -3:3;        % 7 samples around each symbol
    symbol_positions = -3:3;    % 7 neighbor symbols

    max_offset = max(symbol_positions)*step + max(local_window);
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);

    % Preallocate 4D array: [height, width, channels, samples]
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);

    for i = 1:n_valid
        k = valid_range(i);
        if k <= length(symbol_indices)
            current_center = symbol_indices(k);

            % Extract 7x7 matrix
            for r = 1:7
                sym_pos = symbol_positions(r);
                neighbor_center = current_center + sym_pos * step;
                indices = neighbor_center + local_window;

                if all(indices > 0 & indices <= length(rx))
                    X_struct(r, :, 1, i) = real(rx(indices));  % Take real part for BPSK
                end
            end

            y(i) = bits(k);
        end
    end

    % Normalize
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    X_struct = (X_struct - mu) / sig;
end

function net = train_nn(X, y, hidden_sizes, max_epochs, mini_batch)
    % Train fully connected neural network

    % Define layers
    layers = [
        featureInputLayer(size(X,2))
        fullyConnectedLayer(hidden_sizes(1))
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(hidden_sizes(2))
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');

    % Train
    net = trainNetwork(X, categorical(y), layers, options);
end

function net = train_cnn(X_struct, y, max_epochs, mini_batch)
    % Train Structured CNN
    % Input: 7x7x1 image representing ISI structure

    layers = [
        imageInputLayer([7 7 1], 'Normalization', 'none')

        % Conv Layer 1: Process each symbol row independently
        convolution2dLayer([1 7], 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer

        % Conv Layer 2: Combine across symbols (ISI combination)
        convolution2dLayer([7 1], 16, 'Padding', 0)
        batchNormalizationLayer
        reluLayer

        % Classification
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');

    net = trainNetwork(X_struct, categorical(y), layers, options);
end

function bits_hat = detect_fc(X, net)
    % Detect using FC network
    probs = predict(net, X);
    bits_hat = (probs(:,2) > 0.5);  % Column vector to match valid_bits
end

function bits_hat = detect_cnn(X_struct, net)
    % Detect using CNN
    probs = predict(net, X_struct);
    bits_hat = (probs(:,2) > 0.5);  % Column vector to match valid_bits
end
