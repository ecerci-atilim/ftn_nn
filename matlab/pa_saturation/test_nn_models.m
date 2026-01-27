%% TEST_NN_MODELS - Test Trained Neural Networks with File Picker UI
%
% This script allows you to select multiple trained NN models and test them
% over a range of SNR values, generating comparative BER plots.
%
% Usage:
%   1. Run this script
%   2. Use the file picker to select one or more .mat model files
%   3. The script will test all selected models and generate comparison plots
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% FILE SELECTION UI
%% ========================================================================

fprintf('========================================\n');
fprintf('FTN NN Model Testing Suite\n');
fprintf('========================================\n\n');

% Open file picker for multiple selection
[filenames, filepath] = uigetfile('*.mat', ...
    'Select Trained Model Files (Ctrl+Click for multiple)', ...
    'trained_models/*.mat', ...
    'MultiSelect', 'on');

if isequal(filenames, 0)
    fprintf('No files selected. Exiting.\n');
    return;
end

% Convert to cell array if single file selected
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
    fprintf('  Loaded: %s (Type: %s)\n', models{i}.name, models{i}.type);
end
fprintf('\n');

%% ========================================================================
%% TEST CONFIGURATION
%% ========================================================================

% Get configuration from first model
cfg = models{1}.config;

% Test parameters (can be modified)
SNR_test = 0:2:14;          % SNR range for testing
N_test = 50000;             % Symbols per SNR point

fprintf('Test Configuration:\n');
fprintf('  tau = %.2f, beta = %.2f, sps = %d\n', cfg.tau, cfg.beta, cfg.sps);
fprintf('  PA: %s (IBO=%ddB, Enabled=%d)\n', cfg.PA_MODEL, cfg.IBO_dB, cfg.PA_ENABLED);
fprintf('  SNR range: %d to %d dB\n', SNR_test(1), SNR_test(end));
fprintf('  Test symbols: %d per SNR point\n', N_test);
fprintf('\n');

%% ========================================================================
%% SETUP
%% ========================================================================

% Pulse shaping filter
h = rcosdesign(cfg.beta, cfg.span, cfg.sps, 'sqrt');
h = h / norm(h);
delay = cfg.span * cfg.sps;
step = round(cfg.tau * cfg.sps);

% PA parameters
IBO_lin = 10^(cfg.IBO_dB/10);
switch lower(cfg.PA_MODEL)
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
%% TEST ALL MODELS
%% ========================================================================

fprintf('Testing models over SNR range...\n\n');

% Initialize results
results = struct();
for i = 1:length(models)
    results.(matlab.lang.makeValidName(models{i}.name)).BER = zeros(size(SNR_test));
end
results.threshold.BER = zeros(size(SNR_test));

% Print header
header = sprintf('SNR(dB) | Threshold');
for i = 1:length(models)
    header = sprintf('%s | %s', header, models{i}.name);
end
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, length(header)));

% Test each SNR point
for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    rng(100 + snr_idx);
    
    % Generate test data
    bits_test = randi([0 1], 1, N_test);
    [rx_test, sym_idx_test] = generate_ftn_rx(bits_test, cfg.tau, cfg.sps, h, delay, ...
        snr_db, cfg.PA_ENABLED, cfg.PA_MODEL, pa_params);
    
    % Compute margin based on all models
    max_offset = 0;
    for i = 1:length(models)
        if strcmp(models{i}.type, 'fc')
            max_offset = max(max_offset, max(abs(models{i}.offsets)));
        else
            max_offset = max(max_offset, 3*step + 3);
        end
    end
    margin = max_offset + 10;
    valid_range = (margin+1):(N_test-margin);
    
    % Threshold detection
    bits_th = detect_threshold(rx_test, sym_idx_test, valid_range);
    results.threshold.BER(snr_idx) = mean(bits_th ~= bits_test(valid_range));
    
    % Test each model
    for i = 1:length(models)
        mdl = models{i};
        model_name = matlab.lang.makeValidName(mdl.name);
        
        if strcmp(mdl.type, 'fc')
            bits_hat = detect_fc(rx_test, sym_idx_test, valid_range, ...
                mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
        elseif strcmp(mdl.type, 'cnn')
            bits_hat = detect_cnn(rx_test, sym_idx_test, valid_range, ...
                mdl.step, mdl.network);
        end
        
        results.(model_name).BER(snr_idx) = mean(bits_hat ~= bits_test(valid_range));
    end
    
    % Print results for this SNR
    line = sprintf('  %2d    |  %.2e', snr_db, results.threshold.BER(snr_idx));
    for i = 1:length(models)
        model_name = matlab.lang.makeValidName(models{i}.name);
        line = sprintf('%s  |  %.2e', line, results.(model_name).BER(snr_idx));
    end
    fprintf('%s\n', line);
end

%% ========================================================================
%% VISUALIZATION
%% ========================================================================

fprintf('\nGenerating plots...\n');

figure('Position', [100 100 1200 600]);

% Plot 1: BER Curves
subplot(1, 2, 1);
colors = lines(length(models) + 1);
markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*'};

semilogy(SNR_test, results.threshold.BER, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', 'Threshold');
hold on;

for i = 1:length(models)
    model_name = matlab.lang.makeValidName(models{i}.name);
    marker_idx = mod(i-1, length(markers)) + 1;
    semilogy(SNR_test, results.(model_name).BER, ...
        [markers{marker_idx}, '-'], 'Color', colors(i,:), ...
        'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', strrep(models{i}.name, '_', ' '));
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('BER Comparison (\\tau=%.2f, PA=%s)', cfg.tau, cfg.PA_MODEL), 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 9);
ylim([1e-5 1]);

% Plot 2: Performance Summary at 10dB
subplot(1, 2, 2);

% Extract BER at 10dB
snr_10_idx = find(SNR_test == 10, 1);
if isempty(snr_10_idx)
    snr_10_idx = length(SNR_test);  % Use highest SNR if 10dB not in range
end

model_names = cell(length(models), 1);
ber_values = zeros(length(models), 1);
for i = 1:length(models)
    model_names{i} = strrep(models{i}.name, '_', ' ');
    model_name = matlab.lang.makeValidName(models{i}.name);
    ber_values(i) = results.(model_name).BER(snr_10_idx);
end

% Sort by BER (best first)
[ber_sorted, sort_idx] = sort(ber_values);
names_sorted = model_names(sort_idx);
colors_sorted = colors(sort_idx, :);

barh(1:length(models), log10(ber_sorted));
set(gca, 'YTick', 1:length(models), 'YTickLabel', names_sorted);
xlabel('log_{10}(BER)', 'FontSize', 12);
title(sprintf('BER @ %ddB (sorted)', SNR_test(snr_10_idx)), 'FontSize', 13);
grid on;

% Add value labels
for i = 1:length(ber_sorted)
    text(log10(ber_sorted(i)) + 0.1, i, sprintf('%.2e', ber_sorted(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9);
end

sgtitle('FTN NN Model Comparison', 'FontSize', 15, 'FontWeight', 'bold');

% Save figure
[~, name, ~] = fileparts(filenames{1});
saveas(gcf, sprintf('test_results_%s.png', name));

%% ========================================================================
%% SAVE RESULTS
%% ========================================================================

% Save results to .mat file
results_file = sprintf('test_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
save(results_file, 'results', 'SNR_test', 'cfg', 'filenames');
fprintf('\nResults saved to: %s\n', results_file);

fprintf('\n========================================\n');
fprintf('Testing Complete!\n');
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    
    symbols = 2*bits - 1;
    step = round(tau * sps);
    N = length(bits);
    
    tx_up = zeros(N * step, 1);
    tx_up(1:step:end) = symbols;
    tx_shaped = conv(tx_up, h, 'full');
    
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
    else
        tx_pa = tx_shaped;
    end
    
    EbN0 = 10^(SNR_dB/10);
    noise_power = 1 / (2 * EbN0);
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
    
    X_norm = (X - mu) ./ sig;
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
