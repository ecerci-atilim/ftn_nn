%% TRAIN_NN_MODELS - Optimized NN Training for FTN Detection with PA Saturation
%
% This script trains multiple NN architectures with optimizations for best BER:
%   - Corrected noise model and SNR calculation
%   - Decision Feedback (D=4) for improved performance
%   - Hybrid sampling strategy
%   - Multi-SNR training option
%   - Proper normalization handling
%
% Target: BER ~1e-5 at 10dB SNR for tau=0.7
%
% Architectures:
%   FC Variants: Shallow, Standard, Deep, Wide, Bottleneck
%   FC with Decision Feedback: FC_Standard_DF, FC_Hybrid_DF
%   CNN: 1D and 2D variants
%   Recurrent: LSTM, BiLSTM, GRU
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

% Decision Feedback Configuration
D_feedback = 4;             % Feedback depth (use 4 previous decisions)
use_multi_snr = true;       % Train with multiple SNRs for robustness
SNR_train_range = [6, 8, 10];  % SNRs for multi-SNR training

fprintf('========================================\n');
fprintf('FTN NN Training - OPTIMIZED\n');
fprintf('========================================\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
if use_multi_snr
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

% Compute different sampling offsets
offsets = compute_all_offsets(step);

fprintf('Sampling offsets (step=%d):\n', step);
fprintf('  Neighbor: [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:   [%s]\n', num2str(offsets.hybrid));
fprintf('  Fractional: [%s]\n\n', num2str(offsets.fractional));

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/2] Generating training data...\n');
rng(42);

if use_multi_snr
    % Multi-SNR training: generate data at multiple SNRs
    N_per_snr = round(N_train / length(SNR_train_range));
    bits_train_all = [];
    rx_train_all = [];
    sym_idx_offset = 0;
    sym_idx_train_all = [];
    
    for snr_i = 1:length(SNR_train_range)
        snr = SNR_train_range(snr_i);
        bits_temp = randi([0 1], 1, N_per_snr);
        [rx_temp, sym_idx_temp] = generate_ftn_rx_optimized(bits_temp, tau, sps, h, delay, ...
            snr, PA_ENABLED, PA_MODEL, pa_params);
        
        bits_train_all = [bits_train_all, bits_temp];
        rx_train_all = [rx_train_all, rx_temp];
        sym_idx_train_all = [sym_idx_train_all, sym_idx_temp + sym_idx_offset];
        sym_idx_offset = sym_idx_offset + length(rx_temp);
        
        fprintf('  Generated %d symbols @ SNR=%ddB\n', N_per_snr, snr);
    end
    bits_train = bits_train_all;
    rx_train = rx_train_all;
    sym_idx_train = sym_idx_train_all;
else
    bits_train = randi([0 1], 1, N_train);
    [rx_train, sym_idx_train] = generate_ftn_rx_optimized(bits_train, tau, sps, h, delay, ...
        SNR_train, PA_ENABLED, PA_MODEL, pa_params);
end

% Extract features for different sampling strategies
fprintf('  Extracting features...\n');

% Neighbor features (7 inputs)
[X_neighbor, y_neighbor, valid_idx_neighbor] = extract_features_validated(rx_train, bits_train, sym_idx_train, offsets.neighbor);
[X_neighbor_norm, mu_neighbor, sig_neighbor] = normalize_features(X_neighbor);

% Hybrid features (7 inputs, better placement)
[X_hybrid, y_hybrid, valid_idx_hybrid] = extract_features_validated(rx_train, bits_train, sym_idx_train, offsets.hybrid);
[X_hybrid_norm, mu_hybrid, sig_hybrid] = normalize_features(X_hybrid);

% Neighbor + Decision Feedback (7 + D = 11 inputs)
[X_neighbor_df, y_neighbor_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.neighbor, D_feedback);
[X_neighbor_df_norm, mu_neighbor_df, sig_neighbor_df] = normalize_features(X_neighbor_df);

% Hybrid + Decision Feedback (7 + D = 11 inputs)
[X_hybrid_df, y_hybrid_df] = extract_features_with_df(rx_train, bits_train, sym_idx_train, offsets.hybrid, D_feedback);
[X_hybrid_df_norm, mu_hybrid_df, sig_hybrid_df] = normalize_features(X_hybrid_df);

% Structured CNN features (7x7 matrix) with external normalization
[X_cnn2d_raw, y_cnn2d, mu_cnn2d, sig_cnn2d] = extract_structured_features_normalized(rx_train, bits_train, sym_idx_train, step);

fprintf('  Neighbor:     %d samples x %d features\n', size(X_neighbor_norm, 1), size(X_neighbor_norm, 2));
fprintf('  Hybrid:       %d samples x %d features\n', size(X_hybrid_norm, 1), size(X_hybrid_norm, 2));
fprintf('  Neighbor+DF:  %d samples x %d features\n', size(X_neighbor_df_norm, 1), size(X_neighbor_df_norm, 2));
fprintf('  CNN2D:        %dx%dx1x%d\n', size(X_cnn2d_raw, 1), size(X_cnn2d_raw, 2), size(X_cnn2d_raw, 4));
fprintf('\n');

%% ========================================================================
%% PREPARE TRAINING/VALIDATION SPLITS
%% ========================================================================

% Split neighbor data
[X_nb_tr, X_nb_val, y_nb_tr, y_nb_val, idx_nb] = split_data(X_neighbor_norm, y_neighbor, 0.1);

% Split hybrid data
[X_hyb_tr, X_hyb_val, y_hyb_tr, y_hyb_val, ~] = split_data(X_hybrid_norm, y_hybrid, 0.1);

% Split neighbor+DF data
[X_nb_df_tr, X_nb_df_val, y_nb_df_tr, y_nb_df_val, ~] = split_data(X_neighbor_df_norm, y_neighbor_df, 0.1);

% Split hybrid+DF data
[X_hyb_df_tr, X_hyb_df_val, y_hyb_df_tr, y_hyb_df_val, ~] = split_data(X_hybrid_df_norm, y_hybrid_df, 0.1);

% Split CNN2D data (uses same indices proportionally)
n_cnn = size(X_cnn2d_raw, 4);
idx_cnn = randperm(n_cnn);
n_val_cnn = round(0.1 * n_cnn);
X_cnn_val = X_cnn2d_raw(:,:,:,idx_cnn(1:n_val_cnn));
y_cnn_val = categorical(y_cnn2d(idx_cnn(1:n_val_cnn)));
X_cnn_tr = X_cnn2d_raw(:,:,:,idx_cnn(n_val_cnn+1:end));
y_cnn_tr = categorical(y_cnn2d(idx_cnn(n_val_cnn+1:end)));

% Training options with learning rate schedule
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

total_models = 24;  % Updated count with new models
fprintf('[2/2] Training %d architectures...\n\n', total_models);

hWait = waitbar(0, 'Initializing...', 'Name', 'Training NN Models');
training_start = tic;

models = {};
model_idx = 0;

% =========================================================================
% FC VARIANTS - NEIGHBOR SAMPLING (7 inputs)
% =========================================================================

% 1. FC_Shallow
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Shallow', training_start);
fprintf('  [%02d/%02d] FC_Shallow (7->32->2)... ', model_idx, total_models);
tic;
layers = [
    featureInputLayer(7)
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
opts.ValidationData = {X_nb_val, y_nb_val};
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = save_fc_model('FC_Shallow', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->32->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% 2. FC_Standard
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Standard', training_start);
fprintf('  [%02d/%02d] FC_Standard (7->64->32->2)... ', model_idx, total_models);
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
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = save_fc_model('FC_Standard', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->64->32->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% 3. FC_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Deep', training_start);
fprintf('  [%02d/%02d] FC_Deep (7->64->32->16->8->2)... ', model_idx, total_models);
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
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = save_fc_model('FC_Deep', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->64->32->16->8->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% 4. FC_Wide
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Wide', training_start);
fprintf('  [%02d/%02d] FC_Wide (7->256->128->2)... ', model_idx, total_models);
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
net = trainNetwork(X_nb_tr, y_nb_tr, layers, opts);
models{end+1} = save_fc_model('FC_Wide', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->256->128->2', 'sampling', 'neighbor'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% FC WITH HYBRID SAMPLING (7 inputs, better placement)
% =========================================================================

% 5. FC_Hybrid
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Hybrid', training_start);
fprintf('  [%02d/%02d] FC_Hybrid (7->64->32->2, hybrid sampling)... ', model_idx, total_models);
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
opts.ValidationData = {X_hyb_val, y_hyb_val};
net = trainNetwork(X_hyb_tr, y_hyb_tr, layers, opts);
models{end+1} = save_fc_model('FC_Hybrid', net, mu_hybrid, sig_hybrid, offsets.hybrid, ...
    struct('architecture', '7->64->32->2', 'sampling', 'hybrid'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% FC WITH DECISION FEEDBACK (11 inputs = 7 samples + 4 feedback)
% =========================================================================

% 6. FC_Standard_DF (Neighbor + DF)
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Standard_DF', training_start);
fprintf('  [%02d/%02d] FC_Standard_DF (11->64->32->2, neighbor+DF)... ', model_idx, total_models);
tic;
layers = [
    featureInputLayer(7 + D_feedback)  % 11 inputs
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
opts.ValidationData = {X_nb_df_val, y_nb_df_val};
net = trainNetwork(X_nb_df_tr, y_nb_df_tr, layers, opts);
models{end+1} = save_fc_df_model('FC_Standard_DF', net, mu_neighbor_df, sig_neighbor_df, ...
    offsets.neighbor, D_feedback, struct('architecture', '11->64->32->2', 'sampling', 'neighbor', 'df_depth', D_feedback));
fprintf('done (%.1fs)\n', toc);

% 7. FC_Hybrid_DF (Hybrid + DF) - EXPECTED BEST PERFORMER
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Hybrid_DF', training_start);
fprintf('  [%02d/%02d] FC_Hybrid_DF (11->64->32->2, hybrid+DF)... ', model_idx, total_models);
tic;
opts.ValidationData = {X_hyb_df_val, y_hyb_df_val};
net = trainNetwork(X_hyb_df_tr, y_hyb_df_tr, layers, opts);
models{end+1} = save_fc_df_model('FC_Hybrid_DF', net, mu_hybrid_df, sig_hybrid_df, ...
    offsets.hybrid, D_feedback, struct('architecture', '11->64->32->2', 'sampling', 'hybrid', 'df_depth', D_feedback));
fprintf('done (%.1fs)\n', toc);

% 8. FC_Wide_DF (Wide + DF)
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Wide_DF', training_start);
fprintf('  [%02d/%02d] FC_Wide_DF (11->128->64->2, hybrid+DF)... ', model_idx, total_models);
tic;
layers = [
    featureInputLayer(7 + D_feedback)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_hyb_df_tr, y_hyb_df_tr, layers, opts);
models{end+1} = save_fc_df_model('FC_Wide_DF', net, mu_hybrid_df, sig_hybrid_df, ...
    offsets.hybrid, D_feedback, struct('architecture', '11->128->64->2', 'sampling', 'hybrid', 'df_depth', D_feedback));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 1D-CNN VARIANTS
% =========================================================================

X_1d_tr = reshape(X_nb_tr', [7, 1, 1, size(X_nb_tr, 1)]);
X_1d_val = reshape(X_nb_val', [7, 1, 1, size(X_nb_val, 1)]);
opts.ValidationData = {X_1d_val, y_nb_val};

% 9. CNN1D_Simple
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN1D_Simple', training_start);
fprintf('  [%02d/%02d] CNN1D_Simple... ', model_idx, total_models);
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
net = trainNetwork(X_1d_tr, y_nb_tr, layers, opts);
models{end+1} = save_cnn1d_model('CNN1D_Simple', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'Conv3x32->Conv3x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 10. CNN1D_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN1D_Deep', training_start);
fprintf('  [%02d/%02d] CNN1D_Deep... ', model_idx, total_models);
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
net = trainNetwork(X_1d_tr, y_nb_tr, layers, opts);
models{end+1} = save_cnn1d_model('CNN1D_Deep', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'Conv3x32->Conv3x32->Conv3x16->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 2D-CNN VARIANTS (7x7 structured)
% =========================================================================

opts.ValidationData = {X_cnn_val, y_cnn_val};

% 11. CNN2D_Standard
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
models{end+1} = save_cnn2d_model('CNN2D_Standard', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 12. CNN2D_Deep
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
models{end+1} = save_cnn2d_model('CNN2D_Deep', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x32->Conv1x7x32->Conv7x1x16->FC'));
fprintf('done (%.1fs)\n', toc);

% 13. CNN2D_Wide
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
models{end+1} = save_cnn2d_model('CNN2D_Wide', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv1x7x64->Conv7x1x32->FC'));
fprintf('done (%.1fs)\n', toc);

% 14. CNN2D_3x3
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_3x3', training_start);
fprintf('  [%02d/%02d] CNN2D_3x3... ', model_idx, total_models);
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
models{end+1} = save_cnn2d_model('CNN2D_3x3', net, step, mu_cnn2d, sig_cnn2d, ...
    struct('architecture', 'Conv3x3x32->Pool->Conv3x3x64->GAP->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% LSTM/GRU VARIANTS
% =========================================================================

X_seq_tr = permute(reshape(X_nb_tr', [7, 1, size(X_nb_tr, 1)]), [1, 2, 3]);
X_seq_val = permute(reshape(X_nb_val', [7, 1, size(X_nb_val, 1)]), [1, 2, 3]);
X_seq_tr_cell = squeeze(num2cell(X_seq_tr, [1 2]))';
X_seq_val_cell = squeeze(num2cell(X_seq_val, [1 2]))';

opts_seq = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'ValidationData', {X_seq_val_cell, y_nb_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

% 15. LSTM_Simple
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'LSTM_Simple', training_start);
fprintf('  [%02d/%02d] LSTM_Simple... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    lstmLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('LSTM_Simple', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'LSTM32->FC'));
fprintf('done (%.1fs)\n', toc);

% 16. BiLSTM
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'BiLSTM', training_start);
fprintf('  [%02d/%02d] BiLSTM... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    bilstmLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('BiLSTM', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'BiLSTM32->FC'));
fprintf('done (%.1fs)\n', toc);

% 17. LSTM_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'LSTM_Deep', training_start);
fprintf('  [%02d/%02d] LSTM_Deep... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    lstmLayer(32, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(16, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('LSTM_Deep', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'LSTM32->LSTM16->FC'));
fprintf('done (%.1fs)\n', toc);

% 18. GRU_Simple
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'GRU_Simple', training_start);
fprintf('  [%02d/%02d] GRU_Simple... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    gruLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('GRU_Simple', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'GRU32->FC'));
fprintf('done (%.1fs)\n', toc);

% 19. GRU_Deep
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'GRU_Deep', training_start);
fprintf('  [%02d/%02d] GRU_Deep... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    gruLayer(64, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    gruLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('GRU_Deep', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'GRU64->GRU32->FC'));
fprintf('done (%.1fs)\n', toc);

% 20. BiGRU
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'BiGRU', training_start);
fprintf('  [%02d/%02d] BiGRU... ', model_idx, total_models);
tic;
layers = [
    sequenceInputLayer(1)
    bilstmLayer(32, 'OutputMode', 'last')
    dropoutLayer(0.1)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X_seq_tr_cell, y_nb_tr, layers, opts_seq);
models{end+1} = save_lstm_model('BiGRU', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', 'BiGRU32->FC32->FC'));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% DIFFERENT OPTIMIZERS
% =========================================================================

layers_std = [
    featureInputLayer(7)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% 21. FC_SGDM
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_SGDM', training_start);
fprintf('  [%02d/%02d] FC_SGDM... ', model_idx, total_models);
tic;
opts_sgdm = trainingOptions('sgdm', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.01, ...
    'Momentum', 0.9, ...
    'ValidationData', {X_nb_val, y_nb_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');
net = trainNetwork(X_nb_tr, y_nb_tr, layers_std, opts_sgdm);
models{end+1} = save_fc_model('FC_SGDM', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->64->32->2', 'optimizer', 'sgdm'));
fprintf('done (%.1fs)\n', toc);

% 22. FC_RMSProp
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_RMSProp', training_start);
fprintf('  [%02d/%02d] FC_RMSProp... ', model_idx, total_models);
tic;
opts_rmsprop = trainingOptions('rmsprop', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_nb_val, y_nb_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');
net = trainNetwork(X_nb_tr, y_nb_tr, layers_std, opts_rmsprop);
models{end+1} = save_fc_model('FC_RMSProp', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->64->32->2', 'optimizer', 'rmsprop'));
fprintf('done (%.1fs)\n', toc);

% 23. FC_Adam_LRSchedule
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Adam_LRSchedule', training_start);
fprintf('  [%02d/%02d] FC_Adam_LRSchedule... ', model_idx, total_models);
tic;
opts.ValidationData = {X_nb_val, y_nb_val};
net = trainNetwork(X_nb_tr, y_nb_tr, layers_std, opts);
models{end+1} = save_fc_model('FC_Adam_LRSchedule', net, mu_neighbor, sig_neighbor, offsets.neighbor, ...
    struct('architecture', '7->64->32->2', 'optimizer', 'adam+lr_schedule'));
fprintf('done (%.1fs)\n', toc);

% 24. FC_Fractional (fractional sampling)
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'FC_Fractional', training_start);
fprintf('  [%02d/%02d] FC_Fractional... ', model_idx, total_models);
tic;
[X_frac, y_frac, ~] = extract_features_validated(rx_train, bits_train, sym_idx_train, offsets.fractional);
[X_frac_norm, mu_frac, sig_frac] = normalize_features(X_frac);
[X_frac_tr, X_frac_val, y_frac_tr, y_frac_val, ~] = split_data(X_frac_norm, y_frac, 0.1);
opts.ValidationData = {X_frac_val, y_frac_val};
net = trainNetwork(X_frac_tr, y_frac_tr, layers_std, opts);
models{end+1} = save_fc_model('FC_Fractional', net, mu_frac, sig_frac, offsets.fractional, ...
    struct('architecture', '7->64->32->2', 'sampling', 'fractional'));
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
config.use_multi_snr = use_multi_snr;
config.N_train = N_train;
config.max_epochs = max_epochs;
config.mini_batch = mini_batch;
config.D_feedback = D_feedback;
config.timestamp = timestamp;
config.training_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');

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
fprintf('Training Complete!\n');
fprintf('  Models saved: %d\n', length(models));
fprintf('  Total time: %.1f minutes\n', total_time/60);
fprintf('  Output folder: %s\n', output_dir);
fprintf('========================================\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function offsets = compute_all_offsets(step)
    % Compute different sampling strategies
    
    % Neighbor: symbol-rate sampling
    offsets.neighbor = (-3:3) * step;
    
    % Hybrid: center + neighbor instants + inter-symbol (best for ISI)
    t1 = round(step / 3);
    t2 = round(2 * step / 3);
    offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];
    
    % Fractional: spread within inter-symbol space
    offsets.fractional = round((-3:3) * (step-1) / 3);
    offsets.fractional(4) = 0;  % Ensure center is 0
end

function [rx, symbol_indices] = generate_ftn_rx_optimized(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % OPTIMIZED signal generation with corrected noise model
    
    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);
    
    % Optimized upsampling (exact size needed)
    tx_len = 1 + (N-1)*step;
    tx_up = zeros(tx_len, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation (force real output for BPSK)
    if pa_enabled
        tx_pa = real(pa_models(tx_shaped, pa_model, pa_params));
    else
        tx_pa = tx_shaped;
    end
    
    % Corrected SNR calculation: account for actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    
    % Add noise (real for BPSK)
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    
    % Normalize by signal std (not signal+noise)
    rx = rx_mf(:)' / std(rx_mf);
    
    % Symbol indices
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function [X, y, valid_indices] = extract_features_validated(rx, bits, symbol_indices, offsets)
    % Feature extraction with validation
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples);
    y = zeros(n_valid, 1);
    valid_indices = zeros(n_valid, 1);
    actual_valid = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices)
            continue;
        end
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            actual_valid = actual_valid + 1;
            X(actual_valid, :) = real(rx(indices));
            y(actual_valid) = bits(k);
            valid_indices(actual_valid) = k;
        end
    end
    
    % Trim to actual valid samples
    X = X(1:actual_valid, :);
    y = y(1:actual_valid);
    valid_indices = valid_indices(1:actual_valid);
end

function [X, y] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Extract features with decision feedback (uses true bits for training)
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + D + 10;
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples + D);
    y = zeros(n_valid, 1);
    actual_valid = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices) || k <= D
            continue;
        end
        center = symbol_indices(k);
        indices = center + offsets;
        
        if all(indices > 0 & indices <= length(rx))
            actual_valid = actual_valid + 1;
            X(actual_valid, 1:n_samples) = real(rx(indices));
            % Decision feedback: use true previous bits (teacher forcing)
            X(actual_valid, n_samples+1:end) = 2*bits(k-D:k-1) - 1;
            y(actual_valid) = bits(k);
        end
    end
    
    X = X(1:actual_valid, :);
    y = y(1:actual_valid);
end

function [X_struct, y, mu, sig] = extract_structured_features_normalized(rx, bits, symbol_indices, step)
    % Extract 7x7 structured features with external normalization
    N = length(bits);
    local_window = -3:3;
    symbol_positions = -3:3;
    
    max_offset = max(abs(symbol_positions)) * step + max(abs(local_window));
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);
    actual_valid = 0;
    
    for i = 1:n_valid
        k = valid_range(i);
        if k > length(symbol_indices)
            continue;
        end
        current_center = symbol_indices(k);
        
        valid_sample = true;
        temp_matrix = zeros(7, 7);
        
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            
            if all(indices > 0 & indices <= length(rx))
                temp_matrix(r, :) = real(rx(indices));
            else
                valid_sample = false;
                break;
            end
        end
        
        if valid_sample
            actual_valid = actual_valid + 1;
            X_struct(:, :, 1, actual_valid) = temp_matrix;
            y(actual_valid) = bits(k);
        end
    end
    
    X_struct = X_struct(:, :, :, 1:actual_valid);
    y = y(1:actual_valid);
    
    % Compute normalization parameters
    mu = mean(X_struct, 'all');
    sig = std(X_struct, 0, 'all');
    if sig == 0
        sig = 1;
    end
    
    % Apply normalization
    X_struct = (X_struct - mu) / sig;
end

function [X_norm, mu, sig] = normalize_features(X)
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig == 0) = 1;
    X_norm = (X - mu) ./ sig;
end

function [X_tr, X_val, y_tr, y_val, idx] = split_data(X, y, val_ratio)
    n = size(X, 1);
    idx = randperm(n);
    n_val = round(val_ratio * n);
    
    X_val = X(idx(1:n_val), :);
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X(idx(n_val+1:end), :);
    y_tr = categorical(y(idx(n_val+1:end)));
end

function model = save_fc_model(name, net, mu, sig, offsets, info)
    model.name = name;
    model.type = 'fc';
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.info = info;
    model.has_df = false;
end

function model = save_fc_df_model(name, net, mu, sig, offsets, D, info)
    model.name = name;
    model.type = 'fc_df';
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    model.offsets = offsets;
    model.df_depth = D;
    model.info = info;
    model.has_df = true;
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

function model = save_cnn2d_model(name, net, step, mu, sig, info)
    model.name = name;
    model.type = 'cnn2d';
    model.network = net;
    model.step = step;
    model.norm_mu = mu;
    model.norm_sig = sig;
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
