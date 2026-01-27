%% TRAIN_NN_MODELS - Generate Various Trained Neural Networks for FTN Detection
%
% This script trains multiple NN architectures for FTN detection with PA saturation
% and saves them to .mat files for later testing and comparison.
%
% Why Dense Fractional Underperforms:
% -----------------------------------
% 1. CURSE OF DIMENSIONALITY: 15 inputs vs 7 inputs requires exponentially more
%    training data to learn the same mapping quality.
%
% 2. REDUNDANT INFORMATION: T/7 spaced samples are highly correlated (same pulse
%    shape region). More samples ≠ more information, just more noise.
%
% 3. NOISE AMPLIFICATION: Each additional sample adds noise. FC networks can't
%    exploit the correlation structure to filter it.
%
% 4. STRUCTURED CNN WINS: Its architecture matches ISI physics:
%    - Conv(1×7): Process each symbol's local samples (denoising)
%    - Conv(7×1): Combine across symbols (ISI cancellation)
%
% Key Insight: The STRUCTURE of how you process samples matters more than
% the NUMBER of samples. Fractional works when properly structured.
%
% Author: Emre Cerci
% Date: January 2026

clear; clc; close all;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% Output directory for trained models
output_dir = 'trained_models';
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
SNR_train = 10;             % Training SNR
N_train = 100000;           % Training symbols

% Models to train (each entry is a configuration)
models_to_train = {
    % Name, Type, Architecture/Params, Offsets
    struct('name', 'Neighbor_FC_small',    'type', 'fc',  'hidden', [32, 16],    'offset_type', 'neighbor');
    struct('name', 'Neighbor_FC_medium',   'type', 'fc',  'hidden', [64, 32],    'offset_type', 'neighbor');
    struct('name', 'Neighbor_FC_large',    'type', 'fc',  'hidden', [128, 64],   'offset_type', 'neighbor');
    struct('name', 'Neighbor_FC_deep',     'type', 'fc',  'hidden', [64, 32, 16], 'offset_type', 'neighbor');
    struct('name', 'Frac_T2_FC_medium',    'type', 'fc',  'hidden', [64, 32],    'offset_type', 'frac_t2');
    struct('name', 'Frac_T3_FC_medium',    'type', 'fc',  'hidden', [64, 32],    'offset_type', 'frac_t3');
    struct('name', 'Hybrid_FC_medium',     'type', 'fc',  'hidden', [64, 32],    'offset_type', 'hybrid');
    struct('name', 'Struct_CNN_small',     'type', 'cnn', 'filters', [16, 8],    'offset_type', 'structured');
    struct('name', 'Struct_CNN_medium',    'type', 'cnn', 'filters', [32, 16],   'offset_type', 'structured');
    struct('name', 'Struct_CNN_large',     'type', 'cnn', 'filters', [64, 32],   'offset_type', 'structured');
};

fprintf('========================================\n');
fprintf('FTN NN Model Training Suite\n');
fprintf('========================================\n');
fprintf('tau = %.2f, PA: %s (IBO=%ddB)\n', tau, PA_MODEL, IBO_dB);
fprintf('Training: %d symbols @ SNR=%ddB\n', N_train, SNR_train);
fprintf('Models to train: %d\n', length(models_to_train));
fprintf('Output directory: %s\n', output_dir);
fprintf('========================================\n\n');

%% ========================================================================
%% SETUP
%% ========================================================================

% Pulse shaping filter
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;
step = round(tau * sps);

% PA parameters
IBO_lin = 10^(IBO_dB/10);
switch lower(PA_MODEL)
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

% Compute all offset types
offsets = struct();
offsets.neighbor = (-3:3) * step;                           % 7 samples, symbol-rate
offsets.frac_t2 = round((-3:3) * step / 2);                 % 7 samples, T/2 spacing
offsets.frac_t3 = round((-4:4) * step / 3);                 % 9 samples, T/3 spacing
t1 = round(step/3); t2 = round(2*step/3);
offsets.hybrid = [-step, -t2, -t1, 0, t1, t2, step];        % 7 samples, mixed

%% ========================================================================
%% GENERATE TRAINING DATA
%% ========================================================================

fprintf('[1/2] Generating training data...\n');
rng(42);
bits_train = randi([0 1], 1, N_train);
[rx_train, sym_idx_train] = generate_ftn_rx(bits_train, tau, sps, h, delay, ...
    SNR_train, PA_ENABLED, PA_MODEL, pa_params);
fprintf('  Done. rx length = %d samples\n\n', length(rx_train));

%% ========================================================================
%% TRAIN EACH MODEL
%% ========================================================================

fprintf('[2/2] Training models...\n\n');

for m = 1:length(models_to_train)
    cfg = models_to_train{m};
    fprintf('  [%d/%d] Training: %s\n', m, length(models_to_train), cfg.name);
    
    tic;
    
    if strcmp(cfg.type, 'fc')
        % Fully Connected Network
        off = offsets.(cfg.offset_type);
        [X, y] = extract_features(rx_train, bits_train, sym_idx_train, off);
        [X_norm, mu, sig] = normalize_features(X);
        
        net = train_fc_network(X_norm, y, cfg.hidden);
        
        % Save model
        model = struct();
        model.name = cfg.name;
        model.type = 'fc';
        model.network = net;
        model.norm_mu = mu;
        model.norm_sig = sig;
        model.offsets = off;
        model.hidden_sizes = cfg.hidden;
        model.config = struct('tau', tau, 'beta', beta, 'sps', sps, 'span', span, ...
            'PA_MODEL', PA_MODEL, 'IBO_dB', IBO_dB, 'PA_ENABLED', PA_ENABLED, ...
            'SNR_train', SNR_train, 'N_train', N_train);
        
    elseif strcmp(cfg.type, 'cnn')
        % Structured CNN
        [X_struct, y] = extract_structured_features(rx_train, bits_train, sym_idx_train, step);
        
        net = train_cnn_network(X_struct, y, cfg.filters);
        
        % Save model
        model = struct();
        model.name = cfg.name;
        model.type = 'cnn';
        model.network = net;
        model.step = step;
        model.filters = cfg.filters;
        model.config = struct('tau', tau, 'beta', beta, 'sps', sps, 'span', span, ...
            'PA_MODEL', PA_MODEL, 'IBO_dB', IBO_dB, 'PA_ENABLED', PA_ENABLED, ...
            'SNR_train', SNR_train, 'N_train', N_train);
    end
    
    % Save to file
    filename = fullfile(output_dir, sprintf('%s.mat', cfg.name));
    save(filename, 'model', '-v7.3');
    
    elapsed = toc;
    fprintf('        Saved: %s (%.1fs)\n', filename, elapsed);
end

fprintf('\n========================================\n');
fprintf('Training Complete!\n');
fprintf('Models saved to: %s/\n', output_dir);
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

function net = train_fc_network(X, y, hidden_sizes)
    layers = [featureInputLayer(size(X, 2))];
    
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
    
    n = size(X, 1);
    idx = randperm(n);
    n_val = round(0.1 * n);
    X_val = X(idx(1:n_val), :);
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X(idx(n_val+1:end), :);
    y_tr = categorical(y(idx(n_val+1:end)));
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 512, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X_tr, y_tr, layers, options);
end

function net = train_cnn_network(X_struct, y, filters)
    layers = [
        imageInputLayer([7 7 1], 'Normalization', 'none')
        convolution2dLayer([1 7], filters(1), 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([7 1], filters(2), 'Padding', 0)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];
    
    n = size(X_struct, 4);
    idx = randperm(n);
    n_val = round(0.1 * n);
    
    X_val = X_struct(:,:,:,idx(1:n_val));
    y_val = categorical(y(idx(1:n_val)));
    X_tr = X_struct(:,:,:,idx(n_val+1:end));
    y_tr = categorical(y(idx(n_val+1:end)));
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 512, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X_tr, y_tr, layers, options);
end
