%% TRAIN_NN_MODELS - Optimized NN Training for FTN Detection with PA Saturation
%
% This script trains multiple NN architectures with comprehensive optimizations:
%   - Corrected signal processing (noise model, SNR calculation)
%   - Decision Feedback (DF) support for improved BER
%   - Multiple sampling strategies (Neighbor, Hybrid, Fractional)
%   - Consistent normalization across all models
%   - Multi-SNR training for better generalization
%
% Target Performance: BER ~1e-5 @ 10dB SNR with Decision Feedback
%
% Key Optimizations:
%   1. Correct noise power calculation based on actual signal energy
%   2. Decision Feedback (D=4) for FC networks
%   3. Hybrid sampling: captures both symbol-rate and inter-symbol info
%   4. Training at SNR=8dB (challenging region) for better generalization
%   5. Consistent normalization (store parameters for test-time use)
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

% Training Parameters - OPTIMIZED
SNR_train = 8;              % Changed from 10 to 8 for better generalization
N_train = 100000;
max_epochs = 50;
mini_batch = 512;

% Decision Feedback Parameters
D_feedback = 4;             % Feedback depth (use 4 previous decisions)
use_decision_feedback = true;

% Multi-SNR Training (optional, for even better generalization)
use_multi_snr = false;      % Set to true for SNR diversity training
SNR_train_range = [6, 8, 10];

fprintf('========================================\n');
fprintf('FTN NN Architecture Comparison (OPTIMIZED)\n');
fprintf('========================================\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
fprintf('Training: %d symbols @ SNR=%ddB\n', N_train, SNR_train);
fprintf('Decision Feedback: %s (D=%d)\n', mat2str(use_decision_feedback), D_feedback);
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

% Compute different sampling offsets
offsets = struct();

% Neighbor: 7 samples at symbol rate (-3T to +3T)
offsets.neighbor = (-3:3) * step;

% Hybrid: Mix of symbol-rate and inter-symbol (BEST for ISI capture)
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

% Fractional: Spread within ±(step-1), avoids exact symbol instants
offsets.fractional = round((-3:3) * (step-1) / 3);
offsets.fractional(4) = 0;  % Ensure center is at 0

fprintf('Sampling Offsets (step=%d):\n', step);
fprintf('  Neighbor:   [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:     [%s]\n', num2str(offsets.hybrid));
fprintf('  Fractional: [%s]\n\n', num2str(offsets.fractional));

%% ========================================================================
%% GENERATE TRAINING DATA (OPTIMIZED)
%% ========================================================================

fprintf('[1/2] Generating training data...\n');
rng(42);
bits_train = randi([0 1], 1, N_train);

if use_multi_snr
    % Generate data at multiple SNR levels and concatenate
    rx_train = [];
    sym_idx_train = [];
    bits_train_expanded = [];
    
    for snr = SNR_train_range
        [rx_snr, sym_idx_snr] = generate_ftn_rx_optimized(bits_train, tau, sps, h, delay, ...
            snr, PA_ENABLED, PA_MODEL, pa_params);
        rx_train = [rx_train, rx_snr];
        sym_idx_train = [sym_idx_train, sym_idx_snr + length(rx_train) - length(rx_snr)];
        bits_train_expanded = [bits_train_expanded, bits_train];
    end
    bits_train = bits_train_expanded;
    fprintf('  Multi-SNR training: %s dB\n', num2str(SNR_train_range));
else
    [rx_train, sym_idx_train] = generate_ftn_rx_optimized(bits_train, tau, sps, h, delay, ...
        SNR_train, PA_ENABLED, PA_MODEL, pa_params);
end

% Extract features with different sampling strategies

% 1. Neighbor sampling (standard)
[X_neighbor, y_neighbor] = extract_features_optimized(rx_train, bits_train, sym_idx_train, offsets.neighbor);
[X_neighbor_norm, mu_neighbor, sig_neighbor] = normalize_features(X_neighbor);

% 2. Hybrid sampling (better ISI capture)
[X_hybrid, y_hybrid] = extract_features_optimized(rx_train, bits_train, sym_idx_train, offsets.hybrid);
[X_hybrid_norm, mu_hybrid, sig_hybrid] = normalize_features(X_hybrid);

% 3. With Decision Feedback (D=4)
if use_decision_feedback
    [X_neighbor_df, y_neighbor_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, ...
        offsets.neighbor, D_feedback);
    [X_neighbor_df_norm, mu_neighbor_df, sig_neighbor_df] = normalize_features(X_neighbor_df);
    
    [X_hybrid_df, y_hybrid_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, ...
        offsets.hybrid, D_feedback);
    [X_hybrid_df_norm, mu_hybrid_df, sig_hybrid_df] = normalize_features(X_hybrid_df);
end

% 4. Structured 7x7 for CNN
[X_cnn2d, y_cnn2d, mu_cnn2d, sig_cnn2d] = extract_structured_features_optimized(rx_train, bits_train, sym_idx_train, step);

fprintf('  Neighbor input: %d samples x %d features\n', size(X_neighbor_norm, 1), size(X_neighbor_norm, 2));
fprintf('  Hybrid input: %d samples x %d features\n', size(X_hybrid_norm, 1), size(X_hybrid_norm, 2));
if use_decision_feedback
    fprintf('  Neighbor+DF input: %d samples x %d features\n', size(X_neighbor_df_norm, 1), size(X_neighbor_df_norm, 2));
    fprintf('  Hybrid+DF input: %d samples x %d features\n', size(X_hybrid_df_norm, 1), size(X_hybrid_df_norm, 2));
end
fprintf('  CNN2D input: %dx%dx1x%d\n', size(X_cnn2d, 1), size(X_cnn2d, 2), size(X_cnn2d, 4));
fprintf('\n');

%% ========================================================================
%% SPLIT DATA FOR TRAINING/VALIDATION
%% ========================================================================

% Split neighbor data
[X_nb_tr, y_nb_tr, X_nb_val, y_nb_val] = split_data(X_neighbor_norm, y_neighbor, 0.1);

% Split hybrid data
[X_hyb_tr, y_hyb_tr, X_hyb_val, y_hyb_val] = split_data(X_hybrid_norm, y_hybrid, 0.1);

% Split DF data
if use_decision_feedback
    [X_nb_df_tr, y_nb_df_tr, X_nb_df_val, y_nb_df_val] = split_data(X_neighbor_df_norm, y_neighbor_df, 0.1);
    [X_hyb_df_tr, y_hyb_df_tr, X_hyb_df_val, y_hyb_df_val] = split_data(X_hybrid_df_norm, y_hybrid_df, 0.1);
end

% Split CNN data
n_cnn = size(X_cnn2d, 4);
idx_cnn = randperm(n_cnn);
n_val_cnn = round(0.1 * n_cnn);
X_cnn_val = X_cnn2d(:,:,:,idx_cnn(1:n_val_cnn));
y_cnn_val = categorical(y_cnn2d(idx_cnn(1:n_val_cnn)));
X_cnn_tr = X_cnn2d(:,:,:,idx_cnn(n_val_cnn+1:end));
y_cnn_tr = categorical(y_cnn2d(idx_cnn(n_val_cnn+1:end)));

%% ========================================================================
%% TRAINING OPTIONS
%% ========================================================================

opts = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

%% ========================================================================
%% TRAIN ALL ARCHITECTURES
%% ========================================================================

% Count total models
total_models = 14;  % Base models
if use_decision_feedback
    total_models = total_models + 4;  % +4 DF models
end

fprintf('[2/2] Training %d architectures...\n\n', total_models);

% Create progress bar
hWait = waitbar(0, 'Initializing...', 'Name', 'Training NN Models');
training_start = tic;

models = {};
model_idx = 0;

% =========================================================================
% NEIGHBOR SAMPLING MODELS (7 inputs)
% =========================================================================

% 1. Neighbor_FC_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Neighbor_FC', training_start);
fprintf('  [%02d/%02d] Neighbor_FC_Standard... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [64, 32]);
opts.ValidationData = {X_nb_val, y_nb_val};
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = create_model_struct('Neighbor_FC', 'fc', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, struct('architecture', '7->64->32->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% 2. Neighbor_FC_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Neighbor_FC_Deep', training_start);
fprintf('  [%02d/%02d] Neighbor_FC_Deep... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [128, 64, 32]);
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = create_model_struct('Neighbor_FC_Deep', 'fc', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, struct('architecture', '7->128->64->32->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% HYBRID SAMPLING MODELS (7 inputs - better ISI capture)
% =========================================================================

% 3. Hybrid_FC_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_FC', training_start);
fprintf('  [%02d/%02d] Hybrid_FC_Standard... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [64, 32]);
opts.ValidationData = {X_hyb_val, y_hyb_val};
net = trainNetwork(X_hyb_tr, y_hyb_tr, layers, opts);
models{end+1} = create_model_struct('Hybrid_FC', 'fc', net, mu_hybrid, sig_hybrid, ...
    offsets.hybrid, struct('architecture', '7->64->32->2', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% 4. Hybrid_FC_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'Hybrid_FC_Deep', training_start);
fprintf('  [%02d/%02d] Hybrid_FC_Deep... ', model_idx, total_models);
tic;
layers = create_fc_layers(7, [128, 64, 32]);
net = trainNetwork(X_hyb_tr, y_hyb_tr, layers, opts);
models{end+1} = create_model_struct('Hybrid_FC_Deep', 'fc', net, mu_hybrid, sig_hybrid, ...
    offsets.hybrid, struct('architecture', '7->128->64->32->2', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% DECISION FEEDBACK MODELS (7+4=11 inputs) - BEST PERFORMANCE
% =========================================================================

if use_decision_feedback
    % 5. Neighbor_DF_FC
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'Neighbor_DF_FC', training_start);
    fprintf('  [%02d/%02d] Neighbor_DF_FC (7+4 inputs)... ', model_idx, total_models);
    tic;
    layers = create_fc_layers(7 + D_feedback, [64, 32]);
    opts.ValidationData = {X_nb_df_val, y_nb_df_val};
    net = trainNetwork(X_nb_df_tr, y_nb_df_tr, layers, opts);
    models{end+1} = create_model_struct('Neighbor_DF_FC', 'fc_df', net, mu_neighbor_df, sig_neighbor_df, ...
        offsets.neighbor, struct('architecture', '11->64->32->2', 'sampling', 'neighbor', 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
    
    % 6. Neighbor_DF_FC_Deep
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'Neighbor_DF_Deep', training_start);
    fprintf('  [%02d/%02d] Neighbor_DF_FC_Deep... ', model_idx, total_models);
    tic;
    layers = create_fc_layers(7 + D_feedback, [128, 64, 32]);
    net = trainNetwork(X_nb_df_tr, y_nb_df_tr, layers, opts);
    models{end+1} = create_model_struct('Neighbor_DF_Deep', 'fc_df', net, mu_neighbor_df, sig_neighbor_df, ...
        offsets.neighbor, struct('architecture', '11->128->64->32->2', 'sampling', 'neighbor', 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
    
    % 7. Hybrid_DF_FC (EXPECTED BEST)
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'Hybrid_DF_FC', training_start);
    fprintf('  [%02d/%02d] Hybrid_DF_FC (7+4 inputs)... ', model_idx, total_models);
    tic;
    layers = create_fc_layers(7 + D_feedback, [64, 32]);
    opts.ValidationData = {X_hyb_df_val, y_hyb_df_val};
    net = trainNetwork(X_hyb_df_tr, y_hyb_df_tr, layers, opts);
    models{end+1} = create_model_struct('Hybrid_DF_FC', 'fc_df', net, mu_hybrid_df, sig_hybrid_df, ...
        offsets.hybrid, struct('architecture', '11->64->32->2', 'sampling', 'hybrid', 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
    
    % 8. Hybrid_DF_FC_Deep (EXPECTED BEST)
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'Hybrid_DF_Deep', training_start);
    fprintf('  [%02d/%02d] Hybrid_DF_FC_Deep... ', model_idx, total_models);
    tic;
    layers = create_fc_layers(7 + D_feedback, [128, 64, 32]);
    net = trainNetwork(X_hyb_df_tr, y_hyb_df_tr, layers, opts);
    models{end+1} = create_model_struct('Hybrid_DF_Deep', 'fc_df', net, mu_hybrid_df, sig_hybrid_df, ...
        offsets.hybrid, struct('architecture', '11->128->64->32->2', 'sampling', 'hybrid', 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
end

% =========================================================================
% 2D-CNN MODELS (7x7 structured input)
% =========================================================================

opts.ValidationData = {X_cnn_val, y_cnn_val};

% 9. CNN2D_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_Standard', training_start);
fprintf('  [%02d/%02d] CNN2D_Standard... ', model_idx, total_models);
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
models{end+1} = create_cnn2d_model_struct('CNN2D_Standard', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 10. CNN2D_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_Deep', training_start);
fprintf('  [%02d/%02d] CNN2D_Deep... ', model_idx, total_models);
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
models{end+1} = create_cnn2d_model_struct('CNN2D_Deep', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x32->Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 11. CNN2D_Wide
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_Wide', training_start);
fprintf('  [%02d/%02d] CNN2D_Wide... ', model_idx, total_models);
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
models{end+1} = create_cnn2d_model_struct('CNN2D_Wide', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x64->Conv7x1x32->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% LSTM/GRU MODELS (sequence-based)
% =========================================================================

% Reshape for sequence input
X_seq_tr = reshape(X_nb_tr', [7, 1, size(X_nb_tr, 1)]);
X_seq_val = reshape(X_nb_val', [7, 1, size(X_nb_val, 1)]);
X_seq_tr_cell = squeeze(num2cell(X_seq_tr, [1 2]))';
X_seq_val_cell = squeeze(num2cell(X_seq_val, [1 2]))';

opts_seq = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_seq_val_cell, y_nb_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

% 12. LSTM_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'LSTM', training_start);
fprintf('  [%02d/%02d] LSTM_Standard... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    lstmLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = create_model_struct('LSTM_Standard', 'lstm', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, struct('architecture', 'LSTM64->FC32->FC'));
fprintf('done (%.1fs)\n', toc);

% 13. GRU_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'GRU', training_start);
fprintf('  [%02d/%02d] GRU_Standard... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    gruLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = create_model_struct('GRU_Standard', 'lstm', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, struct('architecture', 'GRU64->FC32->FC'));
fprintf('done (%.1fs)\n', toc);

% 14. BiLSTM
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'BiLSTM', training_start);
fprintf('  [%02d/%02d] BiLSTM... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    bilstmLayer(32, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = create_model_struct('BiLSTM', 'lstm', net, mu_neighbor, sig_neighbor, ...
    offsets.neighbor, struct('architecture', 'BiLSTM32->FC'));
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
config.step = step;
config.delay = delay;
config.PA_MODEL = PA_MODEL;
config.IBO_dB = IBO_dB;
config.PA_ENABLED = PA_ENABLED;
config.SNR_train = SNR_train;
config.N_train = N_train;
config.max_epochs = max_epochs;
config.mini_batch = mini_batch;
config.D_feedback = D_feedback;
config.use_decision_feedback = use_decision_feedback;
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
    filename = fullfile(output_dir, sprintf('%s_tau%.1f_SNR%d_%s.mat', ...
        model.name, tau, SNR_train, timestamp));
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

% Close progress bar
close(hWait);
total_time = toc(training_start);

fprintf('\n========================================\n');
fprintf('Training Complete!\n');
fprintf('  Models saved: %d\n', length(models));
fprintf('  Total time: %.1f minutes\n', total_time/60);
fprintf('  Output folder: %s\n', output_dir);
fprintf('  Timestamp: %s\n', timestamp);
fprintf('========================================\n');
fprintf('\nExpected best models for BER ~1e-5 @ 10dB:\n');
fprintf('  1. Hybrid_DF_Deep (hybrid sampling + decision feedback)\n');
fprintf('  2. Hybrid_DF_FC\n');
fprintf('  3. CNN2D_Wide\n');
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
    
    % Optimized upsampling (exact size needed)
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % Apply PA saturation
    if pa_enabled
        tx_pa = pa_models(tx_shaped, pa_model, pa_params);
        % For BPSK, keep signal real after PA
        tx_pa = real(tx_pa);
    else
        tx_pa = tx_shaped;
    end
    
    % OPTIMIZED noise calculation based on actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0_lin = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0_lin);
    
    % Add real noise (BPSK is real-valued)
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    
    % Normalize by standard deviation
    rx = rx_mf(:)' / std(rx_mf);
    
    % Symbol indices
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function [X, y] = extract_features_optimized(rx, bits, symbol_indices, offsets)
    % OPTIMIZED feature extraction with validation
    
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
    
    % Trim to valid samples
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X, y] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Extract features with Decision Feedback
    % Uses true bits as feedback during training (teacher forcing)
    
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + D + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    % Feature size: n_samples + D feedback bits
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
                
                % Signal samples
                X(valid_count, 1:n_samples) = real(rx(indices));
                
                % Decision feedback (true bits for training)
                X(valid_count, n_samples+1:end) = 2*bits(k-D:k-1) - 1;
                
                y(valid_count) = bits(k);
            end
        end
    end
    
    X = X(1:valid_count, :);
    y = y(1:valid_count);
end

function [X_struct, y, mu, sig] = extract_structured_features_optimized(rx, bits, symbol_indices, step)
    % OPTIMIZED structured 7x7 feature extraction with external normalization
    
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
    
    % Compute normalization parameters
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    if sig == 0, sig = 1; end
    
    % Apply normalization
    X_struct = (X_struct - mu) / sig;
end

function [X_norm, mu, sig] = normalize_features(X)
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig == 0) = 1;
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

function model = create_model_struct(name, type, net, mu, sig, offsets, info)
    model.name = name;
    model.type = type;
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.info = info;
end

function model = create_cnn2d_model_struct(name, net, step, mu, sig, info)
    model.name = name;
    model.type = 'cnn2d';
    model.network = net;
    model.step = step;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.info = info;
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
