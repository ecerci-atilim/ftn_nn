%% TRAIN_NN_MODELS_OPTIMIZED - Optimized NN Training for FTN Detection
%
% OPTIMIZATIONS IMPLEMENTED:
% --------------------------
% 1. Fixed noise model: correct SNR calculation based on actual signal power
% 2. Memory optimization: exact upsampling array size
% 3. Validation & error checking in feature extraction
% 4. Consistent normalization between FC and CNN features
% 5. Decision Feedback (D=4) for improved BER
% 6. Hybrid sampling strategy (captures both symbol and inter-symbol info)
% 7. Multi-SNR training for better generalization
% 8. Training SNR changed to 8dB (more challenging = better decision boundaries)
% 9. Numerical safeguards in PA models
% 10. Adaptive testing with minimum error threshold
%
% Target: BER ~1e-5 @ 10dB SNR
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

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
SNR_train = 8;              % Changed from 10 to 8 (more challenging)
SNR_train_range = [6, 8, 10];  % Multi-SNR training
N_train = 100000;
max_epochs = 50;
mini_batch = 512;

% Decision Feedback Parameters
D_feedback = 4;             % Feedback depth (past decisions)
use_decision_feedback = true;

% Sampling Strategies
use_hybrid_sampling = true;

fprintf('========================================\n');
fprintf('FTN NN OPTIMIZED Training\n');
fprintf('========================================\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
fprintf('Training: %d symbols @ SNR=%s dB\n', N_train, mat2str(SNR_train_range));
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

IBO_lin = 10^(IBO_dB/10);
pa_params.G = 1;
pa_params.Asat = sqrt(IBO_lin);
pa_params.p = 2;

% Compute different sampling offsets
offsets = struct();

% Neighbor: 7 samples at symbol rate
offsets.neighbor = (-3:3) * step;

% Hybrid: mix of symbol instants and inter-symbol samples (same 7 samples)
t1 = round(step / 3);
t2 = round(2 * step / 3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];

% Fractional: spread within inter-symbol region
offsets.fractional = round((-3:3) * (step-1) / 3);

fprintf('Sampling offsets (step=%d):\n', step);
fprintf('  Neighbor: [%s]\n', num2str(offsets.neighbor));
fprintf('  Hybrid:   [%s]\n', num2str(offsets.hybrid));
fprintf('  Fractional: [%s]\n\n', num2str(offsets.fractional));

%% ========================================================================
%% GENERATE TRAINING DATA (Multi-SNR)
%% ========================================================================

fprintf('[1/3] Generating training data (multi-SNR)...\n');
rng(42);

% Generate data at multiple SNRs for better generalization
N_per_snr = round(N_train / length(SNR_train_range));
bits_all = [];
rx_all = [];
sym_idx_all = [];

for snr_idx = 1:length(SNR_train_range)
    snr = SNR_train_range(snr_idx);
    fprintf('  Generating %d symbols @ %ddB... ', N_per_snr, snr);
    
    bits_chunk = randi([0 1], 1, N_per_snr);
    [rx_chunk, sym_idx_chunk] = generate_ftn_rx_optimized(bits_chunk, tau, sps, h, delay, ...
        snr, PA_ENABLED, PA_MODEL, pa_params);
    
    % Adjust symbol indices for concatenation
    if ~isempty(rx_all)
        offset_adjust = length(rx_all);
        sym_idx_chunk = sym_idx_chunk + offset_adjust;
    end
    
    bits_all = [bits_all, bits_chunk];
    rx_all = [rx_all, rx_chunk];
    sym_idx_all = [sym_idx_all, sym_idx_chunk];
    
    fprintf('done\n');
end

fprintf('  Total: %d symbols, rx length=%d\n\n', length(bits_all), length(rx_all));

%% ========================================================================
%% EXTRACT FEATURES
%% ========================================================================

fprintf('[2/3] Extracting features...\n');

% Select primary offset type
if use_hybrid_sampling
    primary_offsets = offsets.hybrid;
    fprintf('  Using HYBRID sampling\n');
else
    primary_offsets = offsets.neighbor;
    fprintf('  Using NEIGHBOR sampling\n');
end

% Extract features WITHOUT decision feedback
[X_base, y_base, valid_idx_base] = extract_features_validated(rx_all, bits_all, ...
    sym_idx_all, primary_offsets);
fprintf('  Base features: %d samples x %d features\n', size(X_base, 1), size(X_base, 2));

% Extract features WITH decision feedback
if use_decision_feedback
    [X_df, y_df, valid_idx_df] = extract_features_with_df(rx_all, bits_all, ...
        sym_idx_all, primary_offsets, D_feedback);
    fprintf('  DF features: %d samples x %d features (7 + %d DF)\n', ...
        size(X_df, 1), size(X_df, 2), D_feedback);
end

% Extract structured features for CNN (without internal normalization)
[X_cnn2d_raw, y_cnn2d, valid_idx_cnn] = extract_structured_features_validated(rx_all, ...
    bits_all, sym_idx_all, step);
fprintf('  CNN2D features: %dx%dx1x%d\n', size(X_cnn2d_raw, 1), size(X_cnn2d_raw, 2), size(X_cnn2d_raw, 4));

% Normalize all features consistently
[X_base_norm, mu_base, sig_base] = normalize_features(X_base);
if use_decision_feedback
    [X_df_norm, mu_df, sig_df] = normalize_features(X_df);
end

% Normalize CNN features (flatten, normalize, reshape)
X_cnn_flat = reshape(X_cnn2d_raw, [], size(X_cnn2d_raw, 4))';
[X_cnn_flat_norm, mu_cnn, sig_cnn] = normalize_features(X_cnn_flat);
X_cnn2d = reshape(X_cnn_flat_norm', [7, 7, 1, size(X_cnn2d_raw, 4)]);

fprintf('\n');

%% ========================================================================
%% PREPARE TRAINING/VALIDATION SPLITS
%% ========================================================================

% Split base data
n_base = size(X_base_norm, 1);
idx_base = randperm(n_base);
n_val_base = round(0.1 * n_base);

X_base_val = X_base_norm(idx_base(1:n_val_base), :);
y_base_val = categorical(y_base(idx_base(1:n_val_base)));
X_base_tr = X_base_norm(idx_base(n_val_base+1:end), :);
y_base_tr = categorical(y_base(idx_base(n_val_base+1:end)));

% Split DF data
if use_decision_feedback
    n_df = size(X_df_norm, 1);
    idx_df = randperm(n_df);
    n_val_df = round(0.1 * n_df);
    
    X_df_val = X_df_norm(idx_df(1:n_val_df), :);
    y_df_val = categorical(y_df(idx_df(1:n_val_df)));
    X_df_tr = X_df_norm(idx_df(n_val_df+1:end), :);
    y_df_tr = categorical(y_df(idx_df(n_val_df+1:end)));
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

opts_base = trainingOptions('adam', ...
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
%% TRAIN MODELS
%% ========================================================================

fprintf('[3/3] Training models...\n\n');

total_models = 8;  % Reduced set of best models + DF variants
hWait = waitbar(0, 'Initializing...', 'Name', 'Training Optimized Models');
training_start = tic;

models = {};
model_idx = 0;

% =========================================================================
% 1. FC_Standard (baseline without DF)
% =========================================================================
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

opts_base.ValidationData = {X_base_val, y_base_val};
net = trainNetwork(X_base_tr, y_base_tr, layers, opts_base);
models{end+1} = save_model('FC_Standard', 'fc', net, mu_base, sig_base, primary_offsets, ...
    struct('architecture', '7->64->32->2', 'has_df', false));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 2. FC_Standard_DF (with Decision Feedback)
% =========================================================================
if use_decision_feedback
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'FC_Standard_DF', training_start);
    fprintf('  [%02d/%02d] FC_Standard_DF (11->64->32->2)... ', model_idx, total_models);
    tic;
    
    layers_df = [
        featureInputLayer(7 + D_feedback)  % 7 samples + 4 DF
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
    
    opts_df = opts_base;
    opts_df.ValidationData = {X_df_val, y_df_val};
    net = trainNetwork(X_df_tr, y_df_tr, layers_df, opts_df);
    models{end+1} = save_model('FC_Standard_DF', 'fc_df', net, mu_df, sig_df, primary_offsets, ...
        struct('architecture', '11->64->32->2', 'has_df', true, 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
end

% =========================================================================
% 3. FC_Deep_DF (deeper with DF)
% =========================================================================
if use_decision_feedback
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'FC_Deep_DF', training_start);
    fprintf('  [%02d/%02d] FC_Deep_DF (11->128->64->32->2)... ', model_idx, total_models);
    tic;
    
    layers_deep_df = [
        featureInputLayer(7 + D_feedback)
        fullyConnectedLayer(128)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
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
    
    net = trainNetwork(X_df_tr, y_df_tr, layers_deep_df, opts_df);
    models{end+1} = save_model('FC_Deep_DF', 'fc_df', net, mu_df, sig_df, primary_offsets, ...
        struct('architecture', '11->128->64->32->2', 'has_df', true, 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
end

% =========================================================================
% 4. CNN2D_Standard (baseline)
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_Standard', training_start);
fprintf('  [%02d/%02d] CNN2D_Standard (Conv1x7->Conv7x1->FC)... ', model_idx, total_models);
tic;

layers_cnn = [
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

opts_cnn = opts_base;
opts_cnn.ValidationData = {X_cnn_val, y_cnn_val};
net = trainNetwork(X_cnn_tr, y_cnn_tr, layers_cnn, opts_cnn);
models{end+1} = save_model('CNN2D_Standard', 'cnn2d', net, mu_cnn, sig_cnn, step, ...
    struct('architecture', 'Conv1x7x32->Conv7x1x16->FC', 'has_df', false));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 5. CNN2D_Wide
% =========================================================================
model_idx = model_idx + 1;
update_progress(hWait, model_idx, total_models, 'CNN2D_Wide', training_start);
fprintf('  [%02d/%02d] CNN2D_Wide (Conv1x7x64->Conv7x1x32->FC)... ', model_idx, total_models);
tic;

layers_cnn_wide = [
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

net = trainNetwork(X_cnn_tr, y_cnn_tr, layers_cnn_wide, opts_cnn);
models{end+1} = save_model('CNN2D_Wide', 'cnn2d', net, mu_cnn, sig_cnn, step, ...
    struct('architecture', 'Conv1x7x64->Conv7x1x32->FC', 'has_df', false));
fprintf('done (%.1fs)\n', toc);

% =========================================================================
% 6. GRU_DF (GRU with DF features)
% =========================================================================
if use_decision_feedback
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'GRU_DF', training_start);
    fprintf('  [%02d/%02d] GRU_DF (GRU32->FC)... ', model_idx, total_models);
    tic;
    
    % Reshape DF data for sequence input
    X_seq_df_tr = reshape(X_df_tr', [size(X_df_tr, 2), 1, size(X_df_tr, 1)]);
    X_seq_df_val = reshape(X_df_val', [size(X_df_val, 2), 1, size(X_df_val, 1)]);
    X_seq_df_tr_cell = squeeze(num2cell(X_seq_df_tr, [1 2]))';
    X_seq_df_val_cell = squeeze(num2cell(X_seq_df_val, [1 2]))';
    
    layers_gru_df = [
        sequenceInputLayer(1)
        gruLayer(32, 'OutputMode', 'last')
        dropoutLayer(0.2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    opts_seq = opts_base;
    opts_seq.ValidationData = {X_seq_df_val_cell, y_df_val};
    net = trainNetwork(X_seq_df_tr_cell, y_df_tr, layers_gru_df, opts_seq);
    models{end+1} = save_model('GRU_DF', 'lstm_df', net, mu_df, sig_df, primary_offsets, ...
        struct('architecture', 'GRU32->FC', 'has_df', true, 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
end

% =========================================================================
% 7. BiLSTM_DF
% =========================================================================
if use_decision_feedback
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'BiLSTM_DF', training_start);
    fprintf('  [%02d/%02d] BiLSTM_DF (BiLSTM32->FC)... ', model_idx, total_models);
    tic;
    
    layers_bilstm_df = [
        sequenceInputLayer(1)
        bilstmLayer(32, 'OutputMode', 'last')
        dropoutLayer(0.2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    net = trainNetwork(X_seq_df_tr_cell, y_df_tr, layers_bilstm_df, opts_seq);
    models{end+1} = save_model('BiLSTM_DF', 'lstm_df', net, mu_df, sig_df, primary_offsets, ...
        struct('architecture', 'BiLSTM32->FC', 'has_df', true, 'D', D_feedback));
    fprintf('done (%.1fs)\n', toc);
end

% =========================================================================
% 8. Hybrid_FC_DF (best config: hybrid sampling + DF)
% =========================================================================
if use_decision_feedback
    model_idx = model_idx + 1;
    update_progress(hWait, model_idx, total_models, 'Hybrid_FC_DF', training_start);
    fprintf('  [%02d/%02d] Hybrid_FC_DF (optimal config)... ', model_idx, total_models);
    tic;
    
    % This uses the hybrid offsets already set for X_df
    layers_hybrid = [
        featureInputLayer(7 + D_feedback)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.15)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.15)
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    net = trainNetwork(X_df_tr, y_df_tr, layers_hybrid, opts_df);
    models{end+1} = save_model('Hybrid_FC_DF', 'fc_df', net, mu_df, sig_df, primary_offsets, ...
        struct('architecture', '11->64->64->32->2', 'has_df', true, 'D', D_feedback, ...
               'sampling', 'hybrid'));
    fprintf('done (%.1fs)\n', toc);
end

%% ========================================================================
%% SAVE ALL MODELS
%% ========================================================================

close(hWait);
total_time = toc(training_start);

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
config.N_train = N_train;
config.max_epochs = max_epochs;
config.mini_batch = mini_batch;
config.D_feedback = D_feedback;
config.use_hybrid_sampling = use_hybrid_sampling;
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
    filename = fullfile(output_dir, sprintf('%s_tau%.1f_%s.mat', ...
        model.name, tau, timestamp));
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

fprintf('\n========================================\n');
fprintf('Optimized Training Complete!\n');
fprintf('  Models saved: %d\n', length(models));
fprintf('  Total time: %.1f minutes\n', total_time/60);
fprintf('  Output folder: %s\n', output_dir);
fprintf('========================================\n');

%% ========================================================================
%% OPTIMIZED HELPER FUNCTIONS
%% ========================================================================

function [rx, symbol_indices] = generate_ftn_rx_optimized(bits, tau, sps, h, delay, SNR_dB, ...
    pa_enabled, pa_model, pa_params)
    % OPTIMIZED: Correct SNR calculation, memory optimization
    
    symbols = 2*bits - 1;  % BPSK
    step = round(tau * sps);
    N = length(bits);
    
    % OPTIMIZATION: Exact array size (was N*step, now exact)
    tx_up_len = 1 + (N-1)*step;
    tx_up = zeros(tx_up_len, 1);
    tx_up(1:step:end) = symbols;
    
    % Pulse shaping
    tx_shaped = conv(tx_up, h, 'full');
    
    % PA saturation with numerical safeguards
    if pa_enabled
        tx_pa = pa_models_safe(tx_shaped, pa_model, pa_params);
    else
        tx_pa = tx_shaped;
    end
    
    % OPTIMIZATION: Force real output for BPSK (PA may introduce complex)
    tx_pa = real(tx_pa);
    
    % OPTIMIZATION: Correct noise power based on actual signal power
    signal_power = mean(tx_pa.^2);
    EbN0 = 10^(SNR_dB/10);
    noise_power = signal_power / (2 * EbN0);
    noise = sqrt(noise_power) * randn(size(tx_pa));
    rx_noisy = tx_pa + noise;
    
    % Matched filter
    rx_mf = conv(rx_noisy, h, 'full');
    
    % Normalize by signal std (not signal+noise for consistency)
    rx_mf_signal_only = conv(conv(tx_up, h, 'full'), h, 'full');
    rx = rx_mf(:)' / std(rx_mf_signal_only);
    
    % Symbol indices
    symbol_indices = delay + 1 + (0:N-1) * step;
end

function y = pa_models_safe(x, model_type, params)
    % PA model with numerical safeguards
    
    r = abs(x);
    phi = angle(x);
    
    % Clamp input to prevent overflow
    r = min(r, 100 * params.Asat);
    
    switch lower(model_type)
        case 'rapp'
            r_norm = r / params.Asat;
            r_norm = min(r_norm, 50);  % Prevent overflow in power
            r_out = params.G * r ./ (1 + r_norm.^(2*params.p)).^(1/(2*params.p));
            
        case 'saleh'
            denom = max(1 + params.beta_a * r.^2, 1e-10);  % Prevent div by zero
            r_out = params.alpha_a * r ./ denom;
            phi_out = phi + params.alpha_p * r.^2 ./ (1 + params.beta_p * r.^2);
            phi = phi_out;
            
        case 'soft_limiter'
            r_out = zeros(size(r));
            A_out_sat = params.A_lin + (params.A_sat - params.A_lin) * params.compress;
            
            linear_idx = r <= params.A_lin;
            r_out(linear_idx) = r(linear_idx);
            
            transition_idx = (r > params.A_lin) & (r < params.A_sat);
            r_out(transition_idx) = params.A_lin + ...
                (r(transition_idx) - params.A_lin) * params.compress;
            
            sat_idx = r >= params.A_sat;
            r_out(sat_idx) = A_out_sat;
            
        otherwise
            r_out = r;
    end
    
    y = r_out .* exp(1j * phi);
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
    valid_mask = true(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_range(i);
        
        % Validate symbol index
        if k > length(symbol_indices)
            valid_mask(i) = false;
            continue;
        end
        
        center = symbol_indices(k);
        indices = center + offsets;
        
        % Validate all indices are within bounds
        if ~all(indices > 0 & indices <= length(rx))
            valid_mask(i) = false;
            continue;
        end
        
        X(i, :) = real(rx(indices));
        y(i) = bits(k);
    end
    
    % Remove invalid samples
    X = X(valid_mask, :);
    y = y(valid_mask);
    valid_indices = valid_range(valid_mask);
    
    if sum(~valid_mask) > 0
        fprintf('    Warning: %d invalid samples removed\n', sum(~valid_mask));
    end
end

function [X, y, valid_indices] = extract_features_with_df(rx, bits, symbol_indices, offsets, D)
    % Feature extraction with decision feedback
    % Uses TRUE bits for training (teacher forcing)
    
    N = length(bits);
    n_samples = length(offsets);
    margin = max(abs(offsets)) + D + 10;  % Extra margin for DF
    
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X = zeros(n_valid, n_samples + D);
    y = zeros(n_valid, 1);
    valid_mask = true(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_range(i);
        
        if k > length(symbol_indices) || k <= D
            valid_mask(i) = false;
            continue;
        end
        
        center = symbol_indices(k);
        indices = center + offsets;
        
        if ~all(indices > 0 & indices <= length(rx))
            valid_mask(i) = false;
            continue;
        end
        
        % Signal samples
        X(i, 1:n_samples) = real(rx(indices));
        
        % Decision feedback (use true bits for training)
        X(i, n_samples+1:end) = 2*bits(k-D:k-1) - 1;
        
        y(i) = bits(k);
    end
    
    X = X(valid_mask, :);
    y = y(valid_mask);
    valid_indices = valid_range(valid_mask);
end

function [X_struct, y, valid_indices] = extract_structured_features_validated(rx, bits, symbol_indices, step)
    % Structured feature extraction WITHOUT internal normalization
    
    N = length(bits);
    local_window = -3:3;
    symbol_positions = -3:3;
    
    max_offset = max(abs(symbol_positions)) * step + max(abs(local_window));
    margin = max_offset + 10;
    valid_range = (margin+1):(N-margin);
    n_valid = length(valid_range);
    
    X_struct = zeros(7, 7, 1, n_valid);
    y = zeros(n_valid, 1);
    valid_mask = true(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_range(i);
        
        if k > length(symbol_indices)
            valid_mask(i) = false;
            continue;
        end
        
        current_center = symbol_indices(k);
        row_valid = true;
        
        for r = 1:7
            sym_pos = symbol_positions(r);
            neighbor_center = current_center + sym_pos * step;
            indices = neighbor_center + local_window;
            
            if ~all(indices > 0 & indices <= length(rx))
                row_valid = false;
                break;
            end
            
            X_struct(r, :, 1, i) = real(rx(indices));
        end
        
        if ~row_valid
            valid_mask(i) = false;
            continue;
        end
        
        y(i) = bits(k);
    end
    
    % Remove invalid samples
    X_struct = X_struct(:, :, :, valid_mask);
    y = y(valid_mask);
    valid_indices = valid_range(valid_mask);
end

function [X_norm, mu, sig] = normalize_features(X)
    % Per-feature normalization
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig == 0) = 1;  % Prevent division by zero
    X_norm = (X - mu) ./ sig;
end

function model = save_model(name, type, net, mu, sig, offsets_or_step, info)
    model.name = name;
    model.type = type;
    model.network = net;
    model.norm_mu = mu;
    model.norm_sig = sig;
    
    if strcmp(type, 'cnn2d')
        model.step = offsets_or_step;
    else
        model.offsets = offsets_or_step;
    end
    
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
