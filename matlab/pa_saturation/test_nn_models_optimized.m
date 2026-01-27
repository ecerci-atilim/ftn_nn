%% TEST_NN_MODELS_OPTIMIZED - Optimized Testing with Decision Feedback
%
% OPTIMIZATIONS:
% 1. Sequential detection for models with Decision Feedback
% 2. Adaptive block testing (minimum error threshold)
% 3. Proper normalization using training parameters
% 4. Consistent noise model with training
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% FILE SELECTION UI
%% ========================================================================

fprintf('========================================\n');
fprintf('FTN NN OPTIMIZED Testing Suite\n');
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
    if isfield(models{i}, 'info')
        if isfield(models{i}.info, 'architecture')
            fprintf('      Architecture: %s\n', models{i}.info.architecture);
        end
        if isfield(models{i}.info, 'has_df') && models{i}.info.has_df
            fprintf('      Decision Feedback: D=%d\n', models{i}.info.D);
        end
    end
end
fprintf('\n');

%% ========================================================================
%% TEST CONFIGURATION
%% ========================================================================

cfg = models{1}.config;
SNR_test = 0:2:14;
N_block = 20000;         % Symbols per block
min_errors = 100;        % Minimum errors for reliable BER
max_symbols = 1e6;       % Maximum symbols to test

fprintf('Test Configuration:\n');
fprintf('  tau = %.2f, beta = %.2f, sps = %d\n', cfg.tau, cfg.beta, cfg.sps);
fprintf('  PA: %s (IBO=%ddB)\n', cfg.PA_MODEL, cfg.IBO_dB);
fprintf('  SNR range: %d to %d dB\n', SNR_test(1), SNR_test(end));
fprintf('  Adaptive: min %d errors OR max %.0e symbols\n', min_errors, max_symbols);
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

% Get offsets from config or model
if isfield(cfg, 'offsets') && isfield(cfg.offsets, 'hybrid')
    primary_offsets = cfg.offsets.hybrid;
else
    primary_offsets = (-3:3) * step;
end

%% ========================================================================
%% ADAPTIVE TESTING
%% ========================================================================

fprintf('Testing models (adaptive)...\n\n');

results = struct();
for i = 1:length(models)
    results.(matlab.lang.makeValidName(models{i}.name)).BER = zeros(size(SNR_test));
    results.(matlab.lang.makeValidName(models{i}.name)).errors = zeros(size(SNR_test));
    results.(matlab.lang.makeValidName(models{i}.name)).symbols = zeros(size(SNR_test));
end
results.threshold.BER = zeros(size(SNR_test));

% Print header
fprintf('SNR(dB) | Threshold');
for i = 1:length(models)
    fprintf(' | %s', models{i}.name);
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 12 + 15*length(models)));

for snr_idx = 1:length(SNR_test)
    snr_db = SNR_test(snr_idx);
    
    % Initialize counters for each model
    total_errors = struct();
    total_symbols = struct();
    for i = 1:length(models)
        mname = matlab.lang.makeValidName(models{i}.name);
        total_errors.(mname) = 0;
        total_symbols.(mname) = 0;
    end
    total_errors.threshold = 0;
    total_symbols.threshold = 0;
    
    block_idx = 0;
    
    % Adaptive testing: continue until min_errors reached
    while any_below_threshold(total_errors, min_errors) && ...
          any_below_max(total_symbols, max_symbols)
        
        block_idx = block_idx + 1;
        rng(100*snr_idx + block_idx);
        
        % Generate test data
        bits_test = randi([0 1], 1, N_block);
        [rx_test, sym_idx_test] = generate_ftn_rx_test(bits_test, cfg.tau, cfg.sps, ...
            h, delay, snr_db, cfg.PA_ENABLED, cfg.PA_MODEL, pa_params);
        
        margin = max(abs(primary_offsets)) + 20;
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
            mname = matlab.lang.makeValidName(mdl.name);
            
            if total_errors.(mname) >= min_errors
                continue;  % Already have enough errors
            end
            
            % Detect based on model type
            has_df = isfield(mdl.info, 'has_df') && mdl.info.has_df;
            D = 0;
            if has_df
                D = mdl.info.D;
            end
            
            switch mdl.type
                case 'fc'
                    bits_hat = detect_fc(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'fc_df'
                    bits_hat = detect_fc_df(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig, D);
                    
                case 'cnn1d'
                    bits_hat = detect_cnn1d(rx_test, sym_idx_test, valid_range, ...
                        mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case 'cnn2d'
                    bits_hat = detect_cnn2d(rx_test, sym_idx_test, valid_range, ...
                        mdl.step, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    
                case {'lstm', 'lstm_df'}
                    if has_df
                        bits_hat = detect_lstm_df(rx_test, sym_idx_test, valid_range, ...
                            mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig, D);
                    else
                        bits_hat = detect_lstm(rx_test, sym_idx_test, valid_range, ...
                            mdl.offsets, mdl.network, mdl.norm_mu, mdl.norm_sig);
                    end
            end
            
            total_errors.(mname) = total_errors.(mname) + sum(bits_hat ~= bits_test(valid_range));
            total_symbols.(mname) = total_symbols.(mname) + length(valid_range);
        end
    end
    
    % Calculate BER
    results.threshold.BER(snr_idx) = total_errors.threshold / total_symbols.threshold;
    
    for i = 1:length(models)
        mname = matlab.lang.makeValidName(models{i}.name);
        if total_symbols.(mname) > 0
            results.(mname).BER(snr_idx) = total_errors.(mname) / total_symbols.(mname);
            results.(mname).errors(snr_idx) = total_errors.(mname);
            results.(mname).symbols(snr_idx) = total_symbols.(mname);
        end
    end
    
    % Print row
    fprintf('  %2d    |  %.2e', snr_db, results.threshold.BER(snr_idx));
    for i = 1:length(models)
        mname = matlab.lang.makeValidName(models{i}.name);
        fprintf('  |  %.2e', results.(mname).BER(snr_idx));
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
    mname = matlab.lang.makeValidName(models{i}.name);
    marker_idx = mod(i-1, length(markers)) + 1;
    semilogy(SNR_test, results.(mname).BER, ...
        [markers{marker_idx}, '-'], 'Color', colors(i,:), ...
        'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', strrep(models{i}.name, '_', '\_'));
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('BER Comparison (\\tau=%.2f, PA=%s)', cfg.tau, cfg.PA_MODEL), 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 9);
ylim([1e-6 1]);

% Plot 2: Bar chart at 10dB
subplot(1, 2, 2);
snr_10_idx = find(SNR_test == 10, 1);
if isempty(snr_10_idx), snr_10_idx = length(SNR_test); end

model_names = cell(length(models), 1);
ber_values = zeros(length(models), 1);
for i = 1:length(models)
    model_names{i} = strrep(models{i}.name, '_', ' ');
    mname = matlab.lang.makeValidName(models{i}.name);
    ber_values(i) = results.(mname).BER(snr_10_idx);
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

sgtitle('FTN NN Optimized Detection Performance', 'FontSize', 15, 'FontWeight', 'bold');

% Save
saveas(gcf, sprintf('test_results_optimized_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));

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

% Check if target achieved
target_ber = 1e-5;
achieved = any(ber_sorted <= target_ber);
fprintf('\nTarget BER (1e-5 @ 10dB): %s\n', conditional(achieved, 'ACHIEVED!', 'Not yet achieved'));

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function result = any_below_threshold(errors, threshold)
    fnames = fieldnames(errors);
    result = false;
    for i = 1:length(fnames)
        if errors.(fnames{i}) < threshold
            result = true;
            return;
        end
    end
end

function result = any_below_max(symbols, max_val)
    fnames = fieldnames(symbols);
    result = false;
    for i = 1:length(fnames)
        if symbols.(fnames{i}) < max_val
            result = true;
            return;
        end
    end
end

function [rx, symbol_indices] = generate_ftn_rx_test(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % Generate test signal with same parameters as training
    
    symbols = 2*bits - 1;
    step = round(tau * sps);
    N = length(bits);
    
    tx_up_len = 1 + (N-1)*step;
    tx_up = zeros(tx_up_len, 1);
    tx_up(1:step:end) = symbols;
    
    tx_shaped = conv(tx_up, h, 'full');
    
    if pa_enabled
        tx_pa = pa_models_safe(tx_shaped, pa_model, pa_params);
    else
        tx_pa = tx_shaped;
    end
    
    tx_pa = real(tx_pa);
    
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    rx_mf = conv(rx_noisy, h, 'full');
    rx_mf_signal_only = conv(conv(tx_up, h, 'full'), h, 'full');
    rx = rx_mf(:)' / std(rx_mf_signal_only);
    
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function y = pa_models_safe(x, model_type, params)
    r = abs(x);
    phi = angle(x);
    r = min(r, 100 * params.Asat);
    
    switch lower(model_type)
        case 'rapp'
            r_norm = min(r / params.Asat, 50);
            r_out = params.G * r ./ (1 + r_norm.^(2*params.p)).^(1/(2*params.p));
        otherwise
            r_out = r;
    end
    
    y = r_out .* exp(1j * phi);
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

function bits_hat = detect_fc_df(rx, symbol_indices, valid_range, offsets, net, mu, sig, D)
    % Sequential detection with decision feedback
    n_samples = length(offsets);
    N = max(valid_range) + D;
    bits_hat_full = zeros(1, N);
    
    % Initialize past decisions randomly (or could use threshold)
    margin = min(valid_range) - 1;
    bits_hat_full(1:margin) = randi([0 1], 1, margin);
    
    % Sequential detection
    for idx = 1:length(valid_range)
        k = valid_range(idx);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if ~all(indices > 0 & indices <= length(rx))
            bits_hat_full(k) = randi([0 1]);
            continue;
        end
        
        % Signal features
        x_sig = real(rx(indices));
        
        % Decision feedback features (previous decisions)
        x_df = 2*bits_hat_full(k-D:k-1) - 1;
        
        % Combine and normalize
        x = [x_sig, x_df];
        x_norm = (x - mu) ./ sig;
        
        % Predict
        prob = predict(net, x_norm);
        bits_hat_full(k) = (prob(2) > 0.5);
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
    X_1d = reshape(X_norm', [length(offsets), 1, 1, n_valid]);
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
    
    % Normalize using training parameters
    X_flat = reshape(X_struct, [], n_valid)';
    X_flat_norm = (X_flat - mu) ./ sig;
    X_struct = reshape(X_flat_norm', [7, 7, 1, n_valid]);
    
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

function bits_hat = detect_lstm_df(rx, symbol_indices, valid_range, offsets, net, mu, sig, D)
    % Sequential LSTM detection with decision feedback
    n_samples = length(offsets);
    N = max(valid_range) + D;
    bits_hat_full = zeros(1, N);
    
    margin = min(valid_range) - 1;
    bits_hat_full(1:margin) = randi([0 1], 1, margin);
    
    for idx = 1:length(valid_range)
        k = valid_range(idx);
        center = symbol_indices(k);
        indices = center + offsets;
        
        if ~all(indices > 0 & indices <= length(rx))
            bits_hat_full(k) = randi([0 1]);
            continue;
        end
        
        x_sig = real(rx(indices));
        x_df = 2*bits_hat_full(k-D:k-1) - 1;
        x = [x_sig, x_df];
        x_norm = (x - mu) ./ sig;
        
        x_seq = {x_norm'};
        prob = predict(net, x_seq);
        bits_hat_full(k) = (prob(2) > 0.5);
    end
    
    bits_hat = bits_hat_full(valid_range);
end

function out = conditional(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end
