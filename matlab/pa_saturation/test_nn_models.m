%% TEST_NN_MODELS - Optimized Testing with Decision Feedback Support
%
% Features:
%   - Supports Decision Feedback (DF) models with sequential detection
%   - Adaptive block testing (continues until min_errors reached)
%   - Proper normalization using training parameters
%   - Multiple model types: FC, FC_DF, CNN1D, CNN2D, LSTM
%
% Usage:
%   1. Run this script
%   2. Select model files with file picker (Ctrl+Click for multiple)
%   3. View BER comparison plots and results
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% FILE SELECTION UI
%% ========================================================================

fprintf('========================================\n');
fprintf('FTN NN Model Testing - OPTIMIZED\n');
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
    if isfield(models{i}, 'has_df') && models{i}.has_df
        fprintf('      Decision Feedback: D=%d\n', models{i}.df_depth);
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
max_symbols = 2e6;          % Maximum symbols per SNR point

fprintf('Test Configuration:\n');
fprintf('  tau = %.2f, beta = %.2f, sps = %d\n', cfg.tau, cfg.beta, cfg.sps);
fprintf('  PA: %s (IBO=%ddB)\n', cfg.PA_MODEL, cfg.IBO_dB);
fprintf('  SNR range: %d to %d dB\n', SNR_test(1), SNR_test(end));
fprintf('  Adaptive testing: min %d errors, max %.0e symbols\n', min_errors, max_symbols);
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
%% TEST ALL MODELS WITH ADAPTIVE BLOCK TESTING
%% ========================================================================

fprintf('Testing models (adaptive blocks until %d errors)...\n\n', min_errors);

results = struct();
for i = 1:length(models)
    results.(matlab.lang.makeValidName(models{i}.name)).BER = zeros(size(SNR_test));
    results.(matlab.lang.makeValidName(models{i}.name)).total_errors = zeros(size(SNR_test));
    results.(matlab.lang.makeValidName(models{i}.name)).total_symbols = zeros(size(SNR_test));
end
results.threshold.BER = zeros(size(SNR_test));

% Print header
fprintf('SNR(dB) | Threshold');
for i = 1:min(length(models), 5)  % Show first 5 models in header
    fprintf(' | %s', models{i}.name);
end
if length(models) > 5
    fprintf(' | ...');
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 80));

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    
    % Initialize counters for each model
    model_errors = zeros(length(models), 1);
    model_symbols = zeros(length(models), 1);
    threshold_errors = 0;
    threshold_symbols = 0;
    
    block_idx = 0;
    
    % Continue until all models have enough errors or max symbols reached
    while any(model_errors < min_errors) && any(model_symbols < max_symbols)
        block_idx = block_idx + 1;
        rng(100*snr_idx + block_idx);
        
        % Generate test block
        bits_test = randi([0 1], 1, N_block);
        [rx_test, sym_idx_test] = generate_ftn_rx_optimized(bits_test, cfg.tau, cfg.sps, h, delay, ...
            snr_db, cfg.PA_ENABLED, cfg.PA_MODEL, pa_params);
        
        % Compute margin (use largest offset among all models)
        max_offset = 0;
        max_df = 0;
        for i = 1:length(models)
            mdl = models{i};
            if isfield(mdl, 'offsets')
                max_offset = max(max_offset, max(abs(mdl.offsets)));
            end
            if strcmp(mdl.type, 'cnn2d')
                max_offset = max(max_offset, 3*step + 3);
            end
            if isfield(mdl, 'df_depth')
                max_df = max(max_df, mdl.df_depth);
            end
        end
        margin = max_offset + max_df + 10;
        valid_range = (margin+1):(N_block-margin);
        n_valid = length(valid_range);
        
        % Threshold detection
        if threshold_errors < min_errors
            bits_th = detect_threshold(rx_test, sym_idx_test, valid_range);
            threshold_errors = threshold_errors + sum(bits_th ~= bits_test(valid_range));
            threshold_symbols = threshold_symbols + n_valid;
        end
        
        % Test each model that needs more errors
        for i = 1:length(models)
            if model_errors(i) >= min_errors
                continue;  % This model has enough errors
            end
            
            mdl = models{i};
            
            switch mdl.type
                case 'fc'
                    bits_hat = detect_fc(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'fc_df'
                    % Sequential detection with decision feedback
                    bits_hat = detect_fc_df(rx_test, bits_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig, mdl.df_depth);
                    
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
            
            model_errors(i) = model_errors(i) + sum(bits_hat ~= bits_test(valid_range));
            model_symbols(i) = model_symbols(i) + n_valid;
        end
    end
    
    % Calculate BER
    results.threshold.BER(snr_idx) = threshold_errors / threshold_symbols;
    
    for i = 1:length(models)
        model_name = matlab.lang.makeValidName(models{i}.name);
        if model_symbols(i) > 0
            results.(model_name).BER(snr_idx) = model_errors(i) / model_symbols(i);
        else
            results.(model_name).BER(snr_idx) = NaN;
        end
        results.(model_name).total_errors(snr_idx) = model_errors(i);
        results.(model_name).total_symbols(snr_idx) = model_symbols(i);
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
    
    % Highlight DF models with thicker lines
    if isfield(models{i}, 'has_df') && models{i}.has_df
        lw = 2.5;
    else
        lw = 1.5;
    end
    
    semilogy(SNR_test, results.(model_name).BER, ...
        [markers{marker_idx}, '-'], 'Color', colors(i,:), ...
        'LineWidth', lw, 'MarkerSize', 7, ...
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
    if ~isnan(ber_sorted(i)) && ber_sorted(i) > 0
        text(log10(ber_sorted(i)) + 0.1, i, sprintf('%.2e', ber_sorted(i)), ...
            'VerticalAlignment', 'middle', 'FontSize', 8);
    end
end

sgtitle('FTN NN Architecture Comparison - Optimized', 'FontSize', 15, 'FontWeight', 'bold');

saveas(gcf, sprintf('test_results_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));

%% ========================================================================
%% PRINT SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('SUMMARY @ %ddB SNR (Best to Worst):\n', SNR_test(snr_10_idx));
fprintf('========================================\n');
for i = 1:length(ber_sorted)
    if ~isnan(ber_sorted(i))
        fprintf('  %2d. %-25s : %.2e\n', i, names_sorted{i}, ber_sorted(i));
    end
end
fprintf('========================================\n');

% Highlight best performer
[best_ber, best_idx] = min(ber_values);
fprintf('\nBest performer: %s (BER = %.2e @ 10dB)\n', models{best_idx}.name, best_ber);

% Check if target achieved
if best_ber <= 1e-5
    fprintf('TARGET ACHIEVED: BER <= 1e-5 at 10dB SNR!\n');
elseif best_ber <= 1e-4
    fprintf('Good progress: BER in 1e-4 to 1e-5 range\n');
end

% Save results
results_file = sprintf('test_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
save(results_file, 'results', 'SNR_test', 'cfg', 'filenames');
fprintf('Results saved to: %s\n', results_file);

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_optimized(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    
    symbols = 2*bits - 1;
    step = round(tau * sps);
    N = length(bits);
    
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    tx_shaped = conv(tx_up, h, 'full');
    
    if pa_enabled
        tx_pa = real(pa_models(tx_shaped, pa_model, pa_params));
    else
        tx_pa = tx_shaped;
    end
    
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    
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

function bits_hat = detect_fc_df(rx, bits_init, symbol_indices, valid_range, offsets, net, mu, sig, D)
    % Sequential detection with decision feedback
    % bits_init provides initialization for first D decisions
    
    N = length(symbol_indices);
    bits_hat = zeros(1, N);
    
    % Initialize with true bits for edge symbols
    margin = valid_range(1) - 1;
    bits_hat(1:margin) = bits_init(1:margin);
    
    n_samples = length(offsets);
    
    for i = 1:length(valid_range)
        k = valid_range(i);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx)) && k > D
            % Sample features
            x_samples = real(rx(indices));
            
            % Get previous decisions for feedback
            fb = 2*bits_hat(k-D:k-1) - 1;
            
            % Combine samples + feedback
            x = [x_samples, fb];
            x_norm = (x - mu) ./ sig;
            
            % Predict
            prob = predict(net, x_norm);
            bits_hat(k) = (prob(2) > 0.5);
        end
    end
    
    bits_hat = bits_hat(valid_range);
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
    
    % Use training normalization parameters
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
