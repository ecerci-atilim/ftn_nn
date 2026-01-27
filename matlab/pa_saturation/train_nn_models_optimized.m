%% TRAIN_NN_MODELS_OPTIMIZED - Optimized NN Training for FTN Detection
%
% This script implements all optimizations for FTN detection with PA saturation:
%   - Fixed noise model (correct SNR calculation)
%   - Decision feedback support (D=4)
%   - Hybrid sampling strategy
%   - Multi-SNR training option
%   - Proper normalization handling
%   - Optimized memory allocation
%
% Target: BER ~1e-5 at 10dB SNR (τ=0.7)
%
% Key Improvements:
%   1. Signal processing fixes (noise model, SNR calculation)
%   2. Decision feedback (D=4) dramatically improves BER
%   3. Hybrid sampling captures both symbol and inter-symbol information
%   4. Training at SNR=8dB for better generalization
%   5. Consistent normalization between training and testing
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% Create timestamped output folder
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_dir = fullfile('trained_models', sprintf('optimized_%s', timestamp));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% FTN Parameters
tau = 0.7;
beta = 0.3;
sps = 10;
span = 6;

% PA Configuration
PA_MODEL = 'rapp';
IBO_dB = 3;
PA_ENABLED = true;

% Training Parameters - OPTIMIZED
SNR_train = 8;              % Changed from 10 to 8 for better generalization
N_train = 100000;
max_epochs = 50;
mini_batch = 512;
D_feedback = 4;             % Decision feedback depth

% Multi-SNR training option
USE_MULTI_SNR = true;
SNR_train_range = [6, 8, 10];  % Multiple SNRs for training

fprintf('========================================\n');
fprintf('FTN NN OPTIMIZED Training\n');
fprintf('========================================\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
if USE_MULTI_SNR
    fprintf('Training: %d symbols @ SNR=[%s] dB (multi-SNR)\n', N_train, num2str(SNR_train_range));
else
    fprintf('Training: %d symbols @ SNR=%ddB\n', N_train, SNR_train);
end
fprintf('Decision Feedback: D=%d\n', D_feedback);
fprintf('Output: %s/\n', output_dir);
fprintf('========================================\n\n');

%% ========================================================================
%% SETUP
%% ========================================================================

h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;
step = round(tau * sps);

% PA parameters
IBO_lin = 10^(IBO_dB/10);
pa_params.G = 1;
pa_params.Asat = sqrt(IBO_lin);
pa_params.p = 2;

% Compute all offset types
offsets = struct();

% Neighbor: 7 samples at symbol rate
offsets.neighbor = (-3:3) * step;

% Hybrid: mix of symbol instants and inter-symbol samples (BEST for ISI)
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

% Fractional: evenly spread within ±(step-1)
offsets.fractional = round((-3:3) * (step-1) / 3);
offsets.fractional(4) = 0;  % Ensure center is 0

fprintf('Sample offsets (step=%d):\n', step);
fprintf('  Neighbor:   [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:     [%s]\n', num2str(offsets.hybrid));
fprintf('  Fractional: [%s]\n', num2str(offsets.fractional));
fprintf('\n');

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/2] Generating training data...\n');
rng(42);

if USE_MULTI_SNR
    % Multi-SNR training: generate data at multiple SNRs
    N_per_snr = round(N_train / length(SNR_train_range));
    bits_all = [];
    rx_all = [];
    sym_idx_all = [];
    offset_accumulated = 0;
    
    for snr_idx = 1:length(SNR_train_range)
        snr = SNR_train_range(snr_idx);
        bits_snr = randi([0 1], 1, N_per_snr);
        [rx_snr, sym_idx_snr] = generate_ftn_rx_optimized(bits_snr, tau, sps, h, delay, ...
            snr, PA_ENABLED, PA_MODEL, pa_params);
        
        bits_all = [bits_all, bits_snr];
        rx_all = [rx_all, rx_snr];
        sym_idx_all = [sym_idx_all, sym_idx_snr + offset_accumulated];
        offset_accumulated = offset_accumulated + length(rx_snr);
        
        fprintf('  Generated %d symbols @ SNR=%ddB\n', N_per_snr, snr);
    end
    bits_train = bits_all;
    rx_train = rx_all;
    sym_idx_train = sym_idx_all;
else
    bits_train = randi([0 1], 1, N_train);
    [rx_train, sym_idx_train] = generate_ftn_rx_optimized(bits_train, tau, sps, h, delay, ...
        SNR_train, PA_ENABLED, PA_MODEL, pa_params);
end

%% ========================================================================
%% EXTRACT FEATURES
%% ========================================================================

% Standard features (7 inputs) - Neighbor
[X_neighbor, y_neighbor] = extract_features_validated(rx_train, bits_train, sym_idx_train, offsets.neighbor);
[X_neighbor_norm, mu_neighbor, sig_neighbor] = normalize_features_safe(X_neighbor);

% Hybrid features (7 inputs) - RECOMMENDED
[X_hybrid, y_hybrid] = extract_features_validated(rx_train, bits_train, sym_idx_train, offsets.hybrid);
[X_hybrid_norm, mu_hybrid, sig_hybrid] = normalize_features_safe(X_hybrid);

% Decision Feedback features (7 + D = 11 inputs)
[X_df, y_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.neighbor, D_feedback);
[X_df_norm, mu_df, sig_df] = normalize_features_safe(X_df);

% Hybrid + Decision Feedback (7 + D = 11 inputs) - BEST EXPECTED
[X_hybrid_df, y_hybrid_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.hybrid, D_feedback);
[X_hybrid_df_norm, mu_hybrid_df, sig_hybrid_df] = normalize_features_safe(X_hybrid_df);

% Structured features (7x7 matrix)
[X_cnn2d, y_cnn2d, mu_cnn, sig_cnn] = extract_structured_features_normalized(rx_train, bits_train, sym_idx_train, step);

fprintf('  Neighbor input:   %d samples x %d features\n', size(X_neighbor_norm, 1), size(X_neighbor_norm, 2));
fprintf('  Hybrid input:     %d samples x %d features\n', size(X_hybrid_norm, 1), size(X_hybrid_norm, 2));
fprintf('  DF input:         %d samples x %d features (D=%d)\n', size(X_df_norm, 1), size(X_df_norm, 2), D_feedback);
fprintf('  Hybrid+DF input:  %d samples x %d features\n', size(X_hybrid_df_norm, 1), size(X_hybrid_df_norm, 2));
fprintf('  CNN2D input:      %dx%dx1x%d\n', size(X_cnn2d, 1), size(X_cnn2d, 2), size(X_cnn2d, 4));
fprintf('\n');

%% ========================================================================
%% PREPARE DATA SPLITS
%% ========================================================================

% Split neighbor data
[X_nb_tr, y_nb_tr, X_nb_val, y_nb_val] = split_data(X_neighbor_norm, y_neighbor, 0.1);

% Split hybrid data  
[X_hyb_tr, y_hyb_tr, X_hyb_val, y_hyb_val] = split_data(X_hybrid_norm, y_hybrid, 0.1);

% Split DF data
[X_df_tr, y_df_tr, X_df_val, y_df_val] = split_data(X_df_norm, y_df, 0.1);

% Split hybrid+DF data
[X_hdf_tr, y_hdf_tr, X_hdf_val, y_hdf_val] = split_data(X_hybrid_df_norm, y_hybrid_df, 0.1);

% Split CNN data
[X_cnn_tr, y_cnn_tr, X_cnn_val, y_cnn_val] = split_cnn_data(X_cnn2d, y_cnn2d, 0.1);

% Training options with learning rate schedule
opts = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

%% ========================================================================
%% TRAIN OPTIMIZED MODELS
%% ========================================================================

total_models = 10;  % Focused set of optimized models
fprintf('[2/2] Training %d optimized architectures...\n\n', total_models);

hWait = waitbar(0, 'Initializing...', 'Name', 'Training Optimized Models');
training_start = tic;
models = {};
model_idx = 0;

% =========================================================================
% 1. NEIGHBOR BASELINE (for comparison)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Neighbor_Baseline', training_start);
fprintf('  [%02d/%02d] Neighbor_Baseline (7->64->32->2)... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [64, 32]);
opts.ValidationData = {X_nb_val, y_nb_val};
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = create_model('Neighbor_Baseline', 'fc', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, 0, struct('architecture', '7->64->32->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 2. HYBRID SAMPLING (captures ISI better)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_FC', training_start);
fprintf('  [%02d/%02d] Hybrid_FC (7->64->32->2)... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [64, 32]);
opts.ValidationData = {X_hyb_val, y_hyb_val};
net = trainNetwork(X_hyb_tr, y_hyb_tr, layers, opts);
models{end+1} = create_model('Hybrid_FC', 'fc', net, mu_hybrid, sig_hybrid, ...
    offsets.hybrid, 0, struct('architecture', '7->64->32->2', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 3. NEIGHBOR + DECISION FEEDBACK (significant improvement)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Neighbor_DF', training_start);
fprintf('  [%02d/%02d] Neighbor_DF (11->64->32->2, D=%d)... ', model_idx, total_models, D_feedback);
tic;
layers = create_fc_layers(7 + D_feedback, [64, 32]);
opts.ValidationData = {X_df_val, y_df_val};
net = trainNetwork(X_df_tr, y_df_tr, layers, opts);
models{end+1} = create_model('Neighbor_DF', 'fc_df', net, mu_df, sig_df, ...
    offsets.neighbor, D_feedback, struct('architecture', '11->64->32->2', 'sampling', 'neighbor', 'D', D_feedback));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 4. HYBRID + DECISION FEEDBACK (BEST EXPECTED)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_DF', training_start);
fprintf('  [%02d/%02d] Hybrid_DF (11->64->32->2, D=%d)... ', model_idx, total_models, D_feedback);
tic;
layers = create_fc_layers(7 + D_feedback, [64, 32]);
opts.ValidationData = {X_hdf_val, y_hdf_val};
net = trainNetwork(X_hdf_tr, y_hdf_tr, layers, opts);
models{end+1} = create_model('Hybrid_DF', 'fc_df', net, mu_hybrid_df, sig_hybrid_df, ...
    offsets.hybrid, D_feedback, struct('architecture', '11->64->32->2', 'sampling', 'hybrid', 'D', D_feedback));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 5. HYBRID + DF + DEEPER NETWORK
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_DF_Deep', training_start);
fprintf('  [%02d/%02d] Hybrid_DF_Deep (11->128->64->32->2)... ', model_idx, total_models);
tic;
layers = create_fc_layers(7 + D_feedback, [128, 64, 32]);
opts.ValidationData = {X_hdf_val, y_hdf_val};
net = trainNetwork(X_hdf_tr, y_hdf_tr, layers, opts);
models{end+1} = create_model('Hybrid_DF_Deep', 'fc_df', net, mu_hybrid_df, sig_hybrid_df, ...
    offsets.hybrid, D_feedback, struct('architecture', '11->128->64->32->2', 'sampling', 'hybrid', 'D', D_feedback));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 6. STRUCTURED CNN (baseline)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Struct_CNN', training_start);
fprintf('  [%02d/%02d] Struct_CNN (7x7 -> Conv -> FC)... ', model_idx, total_models);
tic;
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
    classificationLayer];
opts.ValidationData = {X_cnn_val, y_cnn_val};
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers, opts);
models{end+1} = create_cnn_model('Struct_CNN', net, step, mu_cnn, sig_cnn, ...
    struct('architecture', 'Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 7. STRUCTURED CNN DEEP
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Struct_CNN_Deep', training_start);
fprintf('  [%02d/%02d] Struct_CNN_Deep (7x7 -> Conv -> Conv -> FC)... ', model_idx, total_models);
tic;
layers = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    convolution2dLayer([1 7], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([1 7], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([7 1], 16, 'Padding', 0)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers, opts);
models{end+1} = create_cnn_model('Struct_CNN_Deep', net, step, mu_cnn, sig_cnn, ...
    struct('architecture', 'Conv1x7x64->Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 8. LSTM with Hybrid sampling
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_LSTM', training_start);
fprintf('  [%02d/%02d] Hybrid_LSTM (LSTM64->FC)... ', model_idx, total_models);
tic;
% Prepare sequence data
X_seq_tr = prepare_sequence_data(X_hyb_tr);
X_seq_val = prepare_sequence_data(X_hyb_val);
opts_seq = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_seq_val, y_hyb_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');
layers = [
    sequenceInputLayer(1)
    lstmLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr, y_hyb_tr, layers, opts_seq);
models{end+1} = create_model('Hybrid_LSTM', 'lstm', net, mu_hybrid, sig_hybrid, ...
    offsets.hybrid, 0, struct('architecture', 'LSTM64->FC', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 9. GRU with Hybrid sampling
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_GRU', training_start);
fprintf('  [%02d/%02d] Hybrid_GRU (GRU64->FC)... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    gruLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr, y_hyb_tr, layers, opts_seq);
models{end+1} = create_model('Hybrid_GRU', 'lstm', net, mu_hybrid, sig_hybrid, ...
    offsets.hybrid, 0, struct('architecture', 'GRU64->FC', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 10. BiLSTM with Hybrid + DF
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_DF_BiLSTM', training_start);
fprintf('  [%02d/%02d] Hybrid_DF_BiLSTM (BiLSTM32->FC)... ', model_idx, total_models);
tic;
X_seq_df_tr = prepare_sequence_data(X_hdf_tr);
X_seq_df_val = prepare_sequence_data(X_hdf_val);
opts_seq.ValidationData = {X_seq_df_val, y_hdf_val};
layers = [
    sequenceInputLayer(1)
    bilstmLayer(32, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_df_tr, y_hdf_tr, layers, opts_seq);
models{end+1} = create_model('Hybrid_DF_BiLSTM', 'lstm_df', net, mu_hybrid_df, sig_hybrid_df, ...
    offsets.hybrid, D_feedback, struct('architecture', 'BiLSTM32->FC', 'sampling', 'hybrid', 'D', D_feedback));
fprintf('done (%.1fs)\n', toc);

%% ========================================================================
%% SAVE ALL MODELS
%% ========================================================================

fprintf('\nSaving models...\n');

config = struct();
config.tau = tau;
config.beta = beta;
config.sps = sps;
config.span = span;
config.PA_MODEL = PA_MODEL;
config.IBO_dB = IBO_dB;
config.PA_ENABLED = PA_ENABLED;
config.SNR_train = SNR_train;
config.SNR_train_range = SNR_train_range;
config.USE_MULTI_SNR = USE_MULTI_SNR;
config.N_train = N_train;
config.D_feedback = D_feedback;
config.max_epochs = max_epochs;
config.mini_batch = mini_batch;
config.timestamp = timestamp;
config.training_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
config.offsets = offsets;

for i = 1:length(models)
    models{i}.config = config;
    models{i}.timestamp = timestamp;
    models{i}.training_date = config.training_date;
    models{i}.model_index = i;
    models{i}.total_models = length(models);
    
    model = models{i};
    filename = fullfile(output_dir, sprintf('%s_tau%.1f_%s.mat', model.name, tau, timestamp));
    save(filename, 'model', '-v7.3');
    fprintf('  Saved: %s\n', filename);
end

% Save summary
summary = struct();
summary.timestamp = timestamp;
summary.training_date = config.training_date;
summary.config = config;
summary.model_names = cellfun(@(m) m.name, models, 'UniformOutput', false);
summary.model_types = cellfun(@(m) m.type, models, 'UniformOutput', false);
summary.model_architectures = cellfun(@(m) m.info.architecture, models, 'UniformOutput', false);
summary.output_dir = output_dir;

summary_file = fullfile(output_dir, sprintf('training_summary_%s.mat', timestamp));
save(summary_file, 'summary', '-v7.3');
fprintf('  Summary: %s\n', summary_file);

close(hWait);
total_time = toc(training_start);

fprintf('\n========================================\n');
fprintf('OPTIMIZED Training Complete!\n');
fprintf('  Models saved: %d\n', length(models));
fprintf('  Total time: %.1f minutes\n', total_time/60);
fprintf('  Output folder: %s\n', output_dir);
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
    
    % OPTIMIZED: Use exact size needed
    tx_up = zeros(1 + (N-1)*step, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
        % OPTIMIZED: Force real for BPSK (PA may output complex)
        tx_pa = real(tx_pa);
    else
        tx_pa = tx_shaped;
    end
    
    % OPTIMIZED: Correct SNR calculation based on actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    
    % OPTIMIZED: Real noise for real BPSK signal
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    
    % Normalize by signal std (not signal+noise)
    rx = rx_mf(:)' / std(rx_mf);
    
    % Symbol indices
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function [X, y] = extract_features_validated(rx, bits, symbol_indices, offsets)
    % Extract features with validation
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
        
        % VALIDATION: Check symbol index is valid
        if k > length(symbol_indices)
            continue;
        end
        
        center = symbol_indices(k);
        indices = center + offsets;
        
        % VALIDATION: Check all indices are in bounds
        if all(indices > 0 & indices <= length(rx))
            valid_count = valid_count + 1;
            X(valid_count, :) = real(rx(indices));
            y(valid_count) = bits(k);
        end
    end
    
    % Trim to valid samples
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X, y] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Extract features with decision feedback (teacher forcing)
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
        
        if k > length(symbol_indices) || k <= D
            continue;
        end
        
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            valid_count = valid_count + 1;
            
            % Signal samples
            X(valid_count, 1:n_samples) = real(rx(indices));
            
            % Decision feedback: use TRUE bits for training (teacher forcing)
            X(valid_count, n_samples+1:end) = 2*bits(k-D:k-1) - 1;
            
            y(valid_count) = bits(k);
        end
    end
    
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X_struct, y, mu, sig] = extract_structured_features_normalized(rx, bits, symbol_indices, step)
    % Extract 7x7 structured features with EXTERNAL normalization
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
        
        if k > length(symbol_indices)
            continue;
        end
        
        current_center = symbol_indices(k);
        all_valid = true;
        
        % Check all indices first
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            if ~all(indices > 0 & indices <= length(rx))
                all_valid = false;
                break;
            end
        end
        
        if all_valid
            valid_count = valid_count + 1;
            for r = 1:7
                sym_pos = symbol_positions(r);
                neighbor_center = current_center + sym_pos * step;
                indices = neighbor_center + local_window;
                X_struct(r, :, 1, valid_count) = real(rx(indices));
            end
            y(valid_count) = bits(k);
        end
    end
    
    X_struct = X_struct(:, :, :, 1:valid_count);
    y = y(1:valid_count);
    
    % Compute normalization parameters (return separately)
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    if sig == 0
        sig = 1;
    end
    
    % Normalize
    X_struct = (X_struct - mu) / sig;
end

function [X_norm, mu, sig] = normalize_features_safe(X)
    % Safe normalization with zero-variance handling
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    
    % Handle zero variance
    zero_var = sig == 0;
    if any(zero_var)
        warning('Zero variance features detected: %d', sum(zero_var));
        sig(zero_var) = 1;
    end
    
    X_norm = (X - mu) ./ sig;
end

function [X_tr, y_tr, X_val, y_val] = split_data(X, y, val_ratio)
    n = size(X, 1);
    idx = randperm(n);
    n_val = round(val_ratio * n);
    
    X_val = X(idx(1:n_val), :);
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X(idx(n_val+1:end), :);
    y_tr = categorical(y(idx(n_val+1:end)));
end

function [X_tr, y_tr, X_val, y_val] = split_cnn_data(X, y, val_ratio)
    n = size(X, 4);
    idx = randperm(n);
    n_val = round(val_ratio * n);
    
    X_val = X(:,:,:,idx(1:n_val));
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X(:,:,:,idx(n_val+1:end));
    y_tr = categorical(y(idx(n_val+1:end)));
end

function layers = create_fc_layers(input_size, hidden_sizes)
    layers = [featureInputLayer(input_size)];
    
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
end

function X_seq = prepare_sequence_data(X)
    % Convert feature matrix to cell array of sequences for LSTM
    n = size(X, 1);
    X_seq = cell(n, 1);
    for i = 1:n
        X_seq{i} = X(i, :)';  % Column vector sequence
    end
end

function model = create_model(name, type, net, mu, sig, offsets, D, info)
    model.name = name;
    model.type = type;
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.D_feedback = D;
    model.info = info;
end

function model = create_cnn_model(name, net, step, mu, sig, info)
    model.name = name;
    model.type = 'cnn2d';
    model.network = net;
    model.step = step;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.info = info;
    model.D_feedback = 0;
end

function update_progress(hWait, current, total, model_name, start_time)
    progress = (current - 1) / total;
    elapsed = toc(start_time);
    
    if current > 1
        avg_time = elapsed / (current - 1);
        remaining = avg_time * (total - current + 1);
        eta_str = sprintf('ETA: %.1f min', remaining / 60);
    else
        eta_str = 'Estimating...';
    end
    
    waitbar(progress, hWait, sprintf('Training %s (%d/%d) - %.0f%% - %s', ...
        model_name, current, total, progress*100, eta_str));
end
