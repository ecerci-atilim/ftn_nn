%% TRAIN_NN_MODELS - Train Various NN Architectures for FTN Detection
%
% This script trains multiple NN ARCHITECTURES with the SAME input size
% to compare their effectiveness for FTN detection with PA saturation.
%
% All FC models use 7 neighbor samples (symbol-rate)
% All CNN models use 7x7 structured input
%
% Architectures Compared:
% -----------------------
% FC Variants:
%   1. FC_Shallow:     7 -> 32 -> 2 (minimal)
%   2. FC_Standard:    7 -> 64 -> 32 -> 2 (baseline)
%   3. FC_Deep:        7 -> 64 -> 32 -> 16 -> 8 -> 2 (deeper)
%   4. FC_Wide:        7 -> 256 -> 128 -> 2 (wider)
%   5. FC_Bottleneck:  7 -> 128 -> 16 -> 128 -> 2 (compress then expand)
%
% 1D-CNN Variants (treat 7 samples as 1D signal):
%   6. CNN1D_Simple:   Conv1D(3) -> Conv1D(3) -> FC
%   7. CNN1D_Deep:     Conv1D(3) -> Conv1D(3) -> Conv1D(3) -> FC
%
% 2D-CNN Variants (7x7 structured input):
%   8. CNN2D_Standard: Conv(1x7) -> Conv(7x1) -> FC (baseline)
%   9. CNN2D_Deep:     Conv(1x7) -> Conv(1x7) -> Conv(7x1) -> FC
%  10. CNN2D_Parallel: [Conv(1x3) || Conv(1x5) || Conv(1x7)] -> Concat -> Conv(7x1)
%  11. CNN2D_ResNet:   Conv + Skip connections
%
% LSTM/BiLSTM (treat 7 samples as sequence):
%  12. LSTM_Simple:    LSTM(32) -> FC
%  13. BiLSTM:         BiLSTM(32) -> FC
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% Create timestamped output folder
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_dir = fullfile('trained_models', sprintf('run_%s', timestamp));
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

% Training Parameters
SNR_train = 10;
N_train = 100000;
max_epochs = 50;
mini_batch = 512;

fprintf('========================================\n');
fprintf('FTN NN Architecture Comparison\n');
fprintf('========================================\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
fprintf('Training: %d symbols @ SNR=%ddB\n', N_train, SNR_train);
fprintf('Output: %s/\n', output_dir);
fprintf('========================================\n\n');

%% ========================================================================
%% SETUP
%% ========================================================================

h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;
step = round(tau * sps);

IBO_lin = 10^(IBO_dB/10);
pa_params.G = 1;
pa_params.Asat = sqrt(IBO_lin);
pa_params.p = 2;

% Standard offsets: 7 neighbor samples at symbol rate
offsets_neighbor = (-3:3) * step;

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/2] Generating training data...\n');
rng(42);
bits_train = randi([0 1], 1, N_train);
[rx_train, sym_idx_train] = generate_ftn_rx(bits_train, tau, sps, h, delay, ...
    SNR_train, PA_ENABLED, PA_MODEL, pa_params);

% Extract features for FC/1D-CNN/LSTM (7 samples)
[X_fc, y_fc] = extract_features(rx_train, bits_train, sym_idx_train, offsets_neighbor);
[X_fc_norm, mu_fc, sig_fc] = normalize_features(X_fc);

% Extract features for 2D-CNN (7x7 matrix)
[X_cnn2d, y_cnn2d] = extract_structured_features(rx_train, bits_train, sym_idx_train, step);

fprintf('  FC input: %d samples x %d features\n', size(X_fc_norm, 1), size(X_fc_norm, 2));
fprintf('  CNN2D input: %dx%dx1x%d\n', size(X_cnn2d, 1), size(X_cnn2d, 2), size(X_cnn2d, 4));
fprintf('\n');

%% ========================================================================
%% DEFINE MODEL ARCHITECTURES
%% ========================================================================

% Split data for validation
n = size(X_fc_norm, 1);
idx = randperm(n);
n_val = round(0.1 * n);

X_fc_val = X_fc_norm(idx(1:n_val), :);
y_fc_val = categorical(y_fc(idx(1:n_val)));
X_fc_tr = X_fc_norm(idx(n_val+1:end), :);
y_fc_tr = categorical(y_fc(idx(n_val+1:end)));

X_cnn_val = X_cnn2d(:,:,:,idx(1:n_val));
y_cnn_val = categorical(y_cnn2d(idx(1:n_val)));
X_cnn_tr = X_cnn2d(:,:,:,idx(n_val+1:end));
y_cnn_tr = categorical(y_cnn2d(idx(n_val+1:end)));

% Training options
opts = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

%% ========================================================================
%% TRAIN ALL ARCHITECTURES
%% ========================================================================

fprintf('[2/2] Training architectures...\n\n');

models = {};
model_idx = 0;

% =========================================================================
% FC VARIANTS (7 inputs)
% =========================================================================

% 1. FC_Shallow
model_idx = model_idx + 1;
fprintf('  [%02d] FC_Shallow (7->32->2)... ', model_idx);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
opts.ValidationData = {X_fc_val, y_fc_val};
net = trainNetwork(X_fc_tr, y_fc_tr, layers, opts);
models{end+1} = save_fc_model('FC_Shallow', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', '7->32->2'));
fprintf('done (%.1fs)\n', toc);

% 2. FC_Standard
model_idx = model_idx + 1;
fprintf('  [%02d] FC_Standard (7->64->32->2)... ', model_idx);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_fc_tr, y_fc_tr, layers, opts);
models{end+1} = save_fc_model('FC_Standard', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', '7->64->32->2'));
fprintf('done (%.1fs)\n', toc);

% 3. FC_Deep
model_idx = model_idx + 1;
fprintf('  [%02d] FC_Deep (7->64->32->16->8->2)... ', model_idx);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_fc_tr, y_fc_tr, layers, opts);
models{end+1} = save_fc_model('FC_Deep', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', '7->64->32->16->8->2'));
fprintf('done (%.1fs)\n', toc);

% 4. FC_Wide
model_idx = model_idx + 1;
fprintf('  [%02d] FC_Wide (7->256->128->2)... ', model_idx);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_fc_tr, y_fc_tr, layers, opts);
models{end+1} = save_fc_model('FC_Wide', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', '7->256->128->2'));
fprintf('done (%.1fs)\n', toc);

% 5. FC_Bottleneck
model_idx = model_idx + 1;
fprintf('  [%02d] FC_Bottleneck (7->128->16->128->2)... ', model_idx);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)  % Bottleneck
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_fc_tr, y_fc_tr, layers, opts);
models{end+1} = save_fc_model('FC_Bottleneck', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', '7->128->16->128->2'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 1D-CNN VARIANTS (7 inputs treated as 1D signal)
% =========================================================================

% Reshape for 1D CNN: [height=7, width=1, channels=1, samples]
X_1d_tr = reshape(X_fc_tr', [7, 1, 1, size(X_fc_tr, 1)]);
X_1d_val = reshape(X_fc_val', [7, 1, 1, size(X_fc_val, 1)]);
opts.ValidationData = {X_1d_val, y_fc_val};

% 6. CNN1D_Simple
model_idx = model_idx + 1;
fprintf('  [%02d] CNN1D_Simple (Conv3->Conv3->FC)... ', model_idx);
tic;
layers = [
    imageInputLayer([7 1 1], 'Normalization', 'none')
    convolution2dLayer([3 1], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 1], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    flattenLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_1d_tr, y_fc_tr, layers, opts);
models{end+1} = save_cnn1d_model('CNN1D_Simple', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', 'Conv3x32->Conv3x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 7. CNN1D_Deep
model_idx = model_idx + 1;
fprintf('  [%02d] CNN1D_Deep (Conv3->Conv3->Conv3->FC)... ', model_idx);
tic;
layers = [
    imageInputLayer([7 1 1], 'Normalization', 'none')
    convolution2dLayer([3 1], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 1], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 1], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    flattenLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_1d_tr, y_fc_tr, layers, opts);
models{end+1} = save_cnn1d_model('CNN1D_Deep', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', 'Conv3x32->Conv3x32->Conv3x16->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 2D-CNN VARIANTS (7x7 structured input)
% =========================================================================

opts.ValidationData = {X_cnn_val, y_cnn_val};

% 8. CNN2D_Standard
model_idx = model_idx + 1;
fprintf('  [%02d] CNN2D_Standard (Conv1x7->Conv7x1->FC)... ', model_idx);
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
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers, opts);
models{end+1} = save_cnn2d_model('CNN2D_Standard', net, step, ...
    struct('architecture', 'Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 9. CNN2D_Deep
model_idx = model_idx + 1;
fprintf('  [%02d] CNN2D_Deep (Conv1x7->Conv1x7->Conv7x1->FC)... ', model_idx);
tic;
layers = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    convolution2dLayer([1 7], 32, 'Padding', 'same')
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
models{end+1} = save_cnn2d_model('CNN2D_Deep', net, step, ...
    struct('architecture', 'Conv1x7x32->Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 10. CNN2D_Wide
model_idx = model_idx + 1;
fprintf('  [%02d] CNN2D_Wide (Conv1x7x64->Conv7x1x32->FC)... ', model_idx);
tic;
layers = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    convolution2dLayer([1 7], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([7 1], 32, 'Padding', 0)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers, opts);
models{end+1} = save_cnn2d_model('CNN2D_Wide', net, step, ...
    struct('architecture', 'Conv1x7x64->Conv7x1x32->FC'));
fprintf('done (%.1fs)\n', toc);

% 11. CNN2D_MultiScale (different kernel sizes)
model_idx = model_idx + 1;
fprintf('  [%02d] CNN2D_3x3 (Conv3x3->Conv3x3->FC)... ', model_idx);
tic;
layers = [
    imageInputLayer([7 7 1], 'Normalization', 'none')
    convolution2dLayer([3 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', 1, 'Padding', 'same')
    convolution2dLayer([3 3], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers, opts);
models{end+1} = save_cnn2d_model('CNN2D_3x3', net, step, ...
    struct('architecture', 'Conv3x3x32->Pool->Conv3x3x64->GAP->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% LSTM VARIANTS (7 inputs treated as sequence)
% =========================================================================

% Reshape for LSTM: [sequence_length=7, features=1, samples]
X_seq_tr = permute(reshape(X_fc_tr', [7, 1, size(X_fc_tr, 1)]), [1, 2, 3]);
X_seq_val = permute(reshape(X_fc_val', [7, 1, size(X_fc_val, 1)]), [1, 2, 3]);

% Convert to cell array for sequence input
X_seq_tr_cell = squeeze(num2cell(X_seq_tr, [1 2]))';
X_seq_val_cell = squeeze(num2cell(X_seq_val, [1 2]))';

opts_seq = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'ValidationData', {X_seq_val_cell, y_fc_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

% 12. LSTM_Simple
model_idx = model_idx + 1;
fprintf('  [%02d] LSTM_Simple (LSTM32->FC)... ', model_idx);
tic;
layers = [
    sequenceInputLayer(1)
    lstmLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_fc_tr, layers, opts_seq);
models{end+1} = save_lstm_model('LSTM_Simple', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', 'LSTM32->FC'));
fprintf('done (%.1fs)\n', toc);

% 13. BiLSTM
model_idx = model_idx + 1;
fprintf('  [%02d] BiLSTM (BiLSTM32->FC)... ', model_idx);
tic;
layers = [
    sequenceInputLayer(1)
    bilstmLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_fc_tr, layers, opts_seq);
models{end+1} = save_lstm_model('BiLSTM', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', 'BiLSTM32->FC'));
fprintf('done (%.1fs)\n', toc);

% 14. LSTM_Deep
model_idx = model_idx + 1;
fprintf('  [%02d] LSTM_Deep (LSTM32->LSTM16->FC)... ', model_idx);
tic;
layers = [
    sequenceInputLayer(1)
    lstmLayer(32, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(16, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_fc_tr, layers, opts_seq);
models{end+1} = save_lstm_model('LSTM_Deep', net, mu_fc, sig_fc, offsets_neighbor, ...
    struct('architecture', 'LSTM32->LSTM16->FC'));
fprintf('done (%.1fs)\n', toc);

%% ========================================================================
%% SAVE ALL MODELS
%% ========================================================================

fprintf('\nSaving models...\n');

% Comprehensive configuration
config = struct();
config.tau = tau;
config.beta = beta;
config.sps = sps;
config.span = span;
config.PA_MODEL = PA_MODEL;
config.IBO_dB = IBO_dB;
config.PA_ENABLED = PA_ENABLED;
config.SNR_train = SNR_train;
config.N_train = N_train;
config.max_epochs = max_epochs;
config.mini_batch = mini_batch;
config.timestamp = timestamp;
config.training_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');

for i = 1:length(models)
    % Add comprehensive metadata to each model
    models{i}.config = config;
    models{i}.timestamp = timestamp;
    models{i}.training_date = config.training_date;
    models{i}.model_index = i;
    models{i}.total_models = length(models);
    
    model = models{i};
    
    % Filename with timestamp and key parameters
    filename = fullfile(output_dir, sprintf('%s_tau%.1f_SNR%d_%s.mat', ...
        model.name, tau, SNR_train, timestamp));
    save(filename, 'model', '-v7.3');
    fprintf('  Saved: %s\n', filename);
end

% Save a summary file with all model names and configurations
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

fprintf('\n========================================\n');
fprintf('Training Complete!\n');
fprintf('  Models saved: %d\n', length(models));
fprintf('  Output folder: %s\n', output_dir);
fprintf('  Timestamp: %s\n', timestamp);
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

function model = save_fc_model(name, net, mu, sig, offsets, info)
    model.name = name;
    model.type = 'fc';
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.info = info;
end

function model = save_cnn1d_model(name, net, mu, sig, offsets, info)
    model.name = name;
    model.type = 'cnn1d';
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.info = info;
end

function model = save_cnn2d_model(name, net, step, info)
    model.name = name;
    model.type = 'cnn2d';
    model.network = net;
    model.step = step;
    model.info = info;
end

function model = save_lstm_model(name, net, mu, sig, offsets, info)
    model.name = name;
    model.type = 'lstm';
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.info = info;
end
