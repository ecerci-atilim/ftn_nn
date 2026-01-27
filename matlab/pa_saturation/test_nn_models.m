%% TEST_NN_MODELS - Test Trained Neural Networks with File Picker UI
%
% Supports all model types including Decision Feedback:
%   - fc: Fully connected networks
%   - fc_df: FC with Decision Feedback (sequential detection)
%   - cnn1d: 1D convolution
%   - cnn2d: 2D convolution (7x7 structured)
%   - lstm: Recurrent networks (LSTM/GRU/BiLSTM)
%
% Features:
%   - File picker with multiple selection (Ctrl+Click)
%   - Adaptive block testing (min errors threshold)
%   - BER comparison plots
%   - Automatic result saving
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% FILE SELECTION UI
%% ========================================================================

fprintf('========================================\n');
fprintf('FTN NN Model Testing Suite (OPTIMIZED)\n');
fprintf('========================================\n\n');

[filenames, filepath] = uigetfile('*.mat', ...
    'Select Trained Model Files (Ctrl+Click for multiple)', ...
    'trained_models/*.mat', ...
    'MultiSelect', 'on');

if isequal(filenames, 0)
    fprintf('No files selected. Exiting.\n');
    return;
end

if ischar(filenames)
    filenames = {filenames};
end

fprintf('Selected %d model(s):\n', length(filenames));
for i = 1:length(filenames)
    fprintf('  %d. %s\n', i, filenames{i});
end
fprintf('\n');

%% ========================================================================
%% LOAD MODELS
%% ========================================================================

fprintf('Loading models...\n');
models = cell(length(filenames), 1);

for i = 1:length(filenames)
    fullpath = fullfile(filepath, filenames{i});
    data = load(fullpath);
    models{i} = data.model;
    
    fprintf('  [%d] %s\n', i, models{i}.name);
    fprintf('      Type: %s\n', models{i}.type);
    if isfield(models{i}, 'info') && isfield(models{i}.info, 'architecture')
        fprintf('      Architecture: %s\n', models{i}.info.architecture);
    end
    if isfield(models{i}, 'info') && isfield(models{i}.info, 'sampling')
        fprintf('      Sampling: %s\n', models{i}.info.sampling);
    end
    if strcmp(models{i}.type, 'fc_df')
        fprintf('      Decision Feedback: D=%d\n', models{i}.info.D);
    end
    if isfield(models{i}, 'training_date')
        fprintf('      Trained: %s\n', models{i}.training_date);
    end
end
fprintf('\n');

%% ========================================================================
%% TEST CONFIGURATION
%% ========================================================================

cfg = models{1}.config;
SNR_test = 0:2:14;
N_block = 20000;            % Symbols per test block
min_errors = 100;           % Minimum errors for reliable BER
max_symbols = 1e6;          % Maximum symbols to test

fprintf('Test Configuration:\n');
fprintf('  tau = %.2f, beta = %.2f, sps = %d\n', cfg.tau, cfg.beta, cfg.sps);
fprintf('  PA: %s (IBO=%ddB)\n', cfg.PA_MODEL, cfg.IBO_dB);
fprintf('  SNR range: %d to %d dB\n', SNR_test(1), SNR_test(end));
fprintf('  Min errors: %d, Max symbols: %.0e\n', min_errors, max_symbols);
fprintf('\n');

%% ========================================================================
%% SETUP
%% ========================================================================

h = rcosdesign(cfg.beta, cfg.span, cfg.sps, 'sqrt');
h = h / norm(h);
delay = cfg.span * cfg.sps;
step = round(cfg.tau * cfg.sps);

IBO_lin = 10^(cfg.IBO_dB/10);
pa_params.G = 1;
pa_params.Asat = sqrt(IBO_lin);
pa_params.p = 2;

%% ========================================================================
%% TEST ALL MODELS
%% ========================================================================

fprintf('Testing models (adaptive block testing)...\n\n');

results = struct();
for i = 1:length(models)
    results.(matlab.lang.makeValidName(models{i}.name)).BER = zeros(size(SNR_test));
end
results.threshold.BER = zeros(size(SNR_test));

% Print header
fprintf('SNR(dB) | Threshold');
for i = 1:min(length(models), 5)
    fprintf(' | %s', models{i}.name);
end
if length(models) > 5
    fprintf(' | ...');
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 15 + 15*min(length(models), 5)));

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    
    % Initialize error/symbol counters
    total_errors = struct();
    total_symbols = struct();
    for i = 1:length(models)
        name = matlab.lang.makeValidName(models{i}.name);
        total_errors.(name) = 0;
        total_symbols.(name) = 0;
    end
    total_errors.threshold = 0;
    total_symbols.threshold = 0;
    
    block_idx = 0;
    
    % Adaptive testing: continue until min_errors or max_symbols
    while any(structfun(@(x) x < min_errors, total_errors)) && ...
          any(structfun(@(x) x < max_symbols, total_symbols))
        
        block_idx = block_idx + 1;
        rng(100*snr_idx + block_idx);
        
        bits_test = randi([0 1], 1, N_block);
        [rx_test, sym_idx_test] = generate_ftn_rx_test(bits_test, cfg.tau, cfg.sps, h, delay, ...
            snr_db, cfg.PA_ENABLED, cfg.PA_MODEL, pa_params);
        
        % Compute margin
        max_offset = 3 * step + 10;
        margin = max_offset + 10;
        valid_range = (margin+1):(N_block-margin);
        
        % Threshold detection
        if total_errors.threshold < min_errors
            bits_th = detect_threshold(rx_test, sym_idx_test, valid_range);
            total_errors.threshold = total_errors.threshold + sum(bits_th ~= bits_test(valid_range));
            total_symbols.threshold = total_symbols.threshold + length(valid_range);
        end
        
        % Test each model
        for i = 1:length(models)
            mdl = models{i};
            model_name = matlab.lang.makeValidName(mdl.name);
            
            if total_errors.(model_name) >= min_errors
                continue;  % Already have enough errors for this model
            end
            
            switch mdl.type
                case 'fc'
                    bits_hat = detect_fc(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'fc_df'
                    % Sequential detection with decision feedback
                    D = mdl.info.D;
                    bits_hat = detect_fc_df(rx_test, bits_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig, D);
                    
                case 'cnn1d'
                    bits_hat = detect_cnn1d(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'cnn2d'
                    bits_hat = detect_cnn2d(rx_test, sym_idx_test, valid_range, ...
                        mdl.step, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'lstm'
                    bits_hat = detect_lstm(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
            end
            
            total_errors.(model_name) = total_errors.(model_name) + sum(bits_hat ~= bits_test(valid_range));
            total_symbols.(model_name) = total_symbols.(model_name) + length(valid_range);
        end
    end
    
    % Calculate BER
    results.threshold.BER(snr_idx) = total_errors.threshold / total_symbols.threshold;
    for i = 1:length(models)
        model_name = matlab.lang.makeValidName(models{i}.name);
        results.(model_name).BER(snr_idx) = total_errors.(model_name) / total_symbols.(model_name);
    end
    
    % Print row
    fprintf('  %2d    |  %.2e', snr_db, results.threshold.BER(snr_idx));
    for i = 1:min(length(models), 5)
        model_name = matlab.lang.makeValidName(models{i}.name);
        fprintf('  |  %.2e', results.(model_name).BER(snr_idx));
    end
    fprintf('\n');
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

fprintf('\nGenerating plots...\n');

figure('Position', [100 100 1400 600]);

% Plot 1: BER Curves
subplot(1, 2, 1);
colors = lines(length(models) + 1);
markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*', '+', 'x'};

semilogy(SNR_test, results.threshold.BER, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', 'Threshold');
hold on;

for i = 1:length(models)
    model_name = matlab.lang.makeValidName(models{i}.name);
    marker_idx = mod(i-1, length(markers)) + 1;
    semilogy(SNR_test, results.(model_name).BER, ...
        [markers{marker_idx}, '-'], 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 7, ...
        'DisplayName', strrep(models{i}.name, '_', '\_'));
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('BER Comparison (\\tau=%.2f)', cfg.tau), 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 8);
ylim([1e-6 1]);

% Plot 2: Bar chart at 10dB
subplot(1, 2, 2);
snr_10_idx = find(SNR_test == 10, 1);
if isempty(snr_10_idx), snr_10_idx = length(SNR_test); end

model_names = cell(length(models), 1);
ber_values = zeros(length(models), 1);
for i = 1:length(models)
    model_names{i} = strrep(models{i}.name, '_', ' ');
    model_name = matlab.lang.makeValidName(models{i}.name);
    ber_values(i) = results.(model_name).BER(snr_10_idx);
end

[ber_sorted, sort_idx] = sort(ber_values);
names_sorted = model_names(sort_idx);

barh(1:length(models), log10(ber_sorted));
set(gca, 'YTick', 1:length(models), 'YTickLabel', names_sorted, 'FontSize', 9);
xlabel('log_{10}(BER)', 'FontSize', 12);
title(sprintf('BER @ %ddB SNR (Best to Worst)', SNR_test(snr_10_idx)), 'FontSize', 13);
grid on;

for i = 1:length(ber_sorted)
    text(log10(ber_sorted(i)) + 0.1, i, sprintf('%.2e', ber_sorted(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 8);
end

sgtitle('FTN NN Architecture Comparison', 'FontSize', 15, 'FontWeight', 'bold');

saveas(gcf, sprintf('test_results_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));

%% ========================================================================
%% PRINT SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('SUMMARY @ %ddB SNR (Best to Worst):\n', SNR_test(snr_10_idx));
fprintf('========================================\n');
for i = 1:length(ber_sorted)
    fprintf('  %2d. %-25s : %.2e\n', i, names_sorted{i}, ber_sorted(i));
end
fprintf('========================================\n');

% Target check
target_ber = 1e-5;
best_ber = ber_sorted(1);
if best_ber <= target_ber
    fprintf('\n*** TARGET ACHIEVED: Best BER (%.2e) <= 1e-5 ***\n', best_ber);
else
    fprintf('\n Target BER 1e-5 not achieved. Best: %.2e (%.1fx off)\n', best_ber, best_ber/target_ber);
end

% Save results
results_file = sprintf('test_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
save(results_file, 'results', 'SNR_test', 'cfg', 'filenames');
fprintf('Results saved to: %s\n', results_file);

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_test(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % OPTIMIZED signal generation (same as training)
    
    symbols = 2*bits - 1;
    step = round(tau * sps);
    N = length(bits);
    
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    tx_shaped = conv(tx_up, h, 'full');
    
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
        tx_pa = real(tx_pa);
    else
        tx_pa = tx_shaped;
    end
    
    signal_power = mean(tx_pa.^2);
    EbN0_lin = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0_lin);
    
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    rx_mf = conv(rx_noisy, h, 'full');
    rx = rx_mf(:)' / std(rx_mf);
    symbol_indices = delay + 1 + (0:N-1) * step;
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

function bits_hat = detect_fc(rx, symbol_indices, valid_range, offsets, net, mu, sig)
    n_valid = length(valid_range);
    X = zeros(n_valid, length(offsets));
    
    for i = 1:n_valid
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        if all(indices > 0 & indices <= length(rx))
            X(i, :) = real(rx(indices));
        end
    end
    
    X_norm = (X - mu) ./ sig;
    probs = predict(net, X_norm);
    bits_hat = (probs(:, 2) > 0.5)';
end

function bits_hat = detect_fc_df(rx, bits_true, symbol_indices, valid_range, offsets, net, mu, sig, D)
    % Sequential detection with decision feedback
    % Uses previous DECISIONS (not true bits) for feedback
    
    N = max(valid_range);
    bits_hat_full = zeros(1, N);
    
    % Initialize with random bits for the first D positions
    margin = valid_range(1) - 1;
    bits_hat_full(1:margin) = randi([0 1], 1, margin);
    
    n_samples = length(offsets);
    
    for i = 1:length(valid_range)
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx)) && k > D
            % Signal samples
            x_signal = real(rx(indices));
            
            % Decision feedback (previous decisions, not true bits)
            x_feedback = 2*bits_hat_full(k-D:k-1) - 1;
            
            % Combine
            x = [x_signal, x_feedback];
            x_norm = (x - mu) ./ sig;
            
            % Predict
            prob = predict(net, x_norm);
            bits_hat_full(k) = (prob(2) > 0.5);
        end
    end
    
    bits_hat = bits_hat_full(valid_range);
end

function bits_hat = detect_cnn1d(rx, symbol_indices, valid_range, offsets, net, mu, sig)
    n_valid = length(valid_range);
    X = zeros(n_valid, length(offsets));
    
    for i = 1:n_valid
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        if all(indices > 0 & indices <= length(rx))
            X(i, :) = real(rx(indices));
        end
    end
    
    X_norm = (X - mu) ./ sig;
    X_1d = reshape(X_norm', [7, 1, 1, n_valid]);
    probs = predict(net, X_1d);
    bits_hat = (probs(:, 2) > 0.5)';
end

function bits_hat = detect_cnn2d(rx, symbol_indices, valid_range, step, net, mu, sig)
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
    
    % Use stored normalization parameters
    X_struct = (X_struct - mu) / sig;
    
    probs = predict(net, X_struct);
    bits_hat = (probs(:, 2) > 0.5)';
end

function bits_hat = detect_lstm(rx, symbol_indices, valid_range, offsets, net, mu, sig)
    n_valid = length(valid_range);
    X = zeros(n_valid, length(offsets));
    
    for i = 1:n_valid
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        if all(indices > 0 & indices <= length(rx))
            X(i, :) = real(rx(indices));
        end
    end
    
    X_norm = (X - mu) ./ sig;
    
    X_seq = cell(n_valid, 1);
    for i = 1:n_valid
        X_seq{i} = X_norm(i, :)';
    end
    
    probs = predict(net, X_seq);
    bits_hat = (probs(:, 2) > 0.5)';
end
