%% FTN NN Training - CORRECTED SNR
% SNR tanımı düzeltildi
% Power normalization opsiyonel
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% ========== PARAMETERS ==========
tau = 0.7;
window_len = 31;
num_feedback = 4;
num_neighbor = 3;
SNR_train_dB = 8;
N_symbols = 100000;
teacher_forcing_ratio = 0.7;

% SNR method: 'corrected' or 'old'
SNR_METHOD = 'corrected';

fprintf('=== Training Models for τ=%.1f ===\n', tau);
fprintf('SNR Method: %s\n\n', SNR_METHOD);

%% ========== TRAIN MODELS ==========

% Model 1: Single
fprintf('Model 1: Single\n');
[X1, Y1] = generate_data_corrected(N_symbols, tau, SNR_train_dB, 1, 0, 0, 0, false, SNR_METHOD);
net1 = train_model(X1, Y1);

% Model 2: Single+DF
fprintf('Model 2: Single+DF\n');
[X2, Y2] = generate_data_corrected(N_symbols, tau, SNR_train_dB, 1, num_feedback, teacher_forcing_ratio, 0, false, SNR_METHOD);
net2 = train_model(X2, Y2);

% Model 3: Neighbors
fprintf('Model 3: Neighbors\n');
[X3, Y3] = generate_data_corrected(N_symbols, tau, SNR_train_dB, 1, 0, 0, num_neighbor, true, SNR_METHOD);
net3 = train_model(X3, Y3);

% Model 4: Neighbors+DF
fprintf('Model 4: Neighbors+DF\n');
[X4, Y4] = generate_data_corrected(N_symbols, tau, SNR_train_dB, 1, num_feedback, teacher_forcing_ratio, num_neighbor, true, SNR_METHOD);
net4 = train_model(X4, Y4);

% Model 5: Window
fprintf('Model 5: Window\n');
[X5, Y5] = generate_data_corrected(N_symbols, tau, SNR_train_dB, window_len, 0, 0, 0, false, SNR_METHOD);
net5 = train_model(X5, Y5);

% Model 6: Window+DF
fprintf('Model 6: Window+DF\n');
[X6, Y6] = generate_data_corrected(N_symbols, tau, SNR_train_dB, window_len, num_feedback, teacher_forcing_ratio, 0, false, SNR_METHOD);
net6 = train_model(X6, Y6);

% Model 7: Full
fprintf('Model 7: Full\n');
[X7, Y7] = generate_data_corrected(N_symbols, tau, SNR_train_dB, window_len, num_feedback, teacher_forcing_ratio, num_neighbor, false, SNR_METHOD);
net7 = train_model(X7, Y7);

%% ========== SAVE ==========
if ~exist('mat/comparison', 'dir'), mkdir('mat/comparison'); end
fname = sprintf('mat/comparison/tau%02d_snr_corrected.mat', tau*10);
save(fname, 'net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', ...
    'tau', 'window_len', 'num_feedback', 'num_neighbor', 'SNR_METHOD');
fprintf('\nSaved: %s\n', fname);

%% ========== FUNCTIONS ==========

function [X, Y] = generate_data_corrected(N, tau, SNR_dB, win_len, num_fb, tf_ratio, num_nb, neighbors_only, snr_method)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    bits = randi([0 1], N, 1);
    symbols = 2 * bits - 1;
    
    % Transmit
    tx = conv(upsample(symbols, step), h);
    
    % CORRECTED SNR calculation
    EbN0_linear = 10^(SNR_dB/10);
    if strcmp(snr_method, 'corrected')
        sig_pwr = mean(tx.^2);
        noise_var = sig_pwr / (2 * EbN0_linear);
    else
        noise_var = 1 / (2 * EbN0_linear);  % old method
    end
    
    rx = tx + sqrt(noise_var) * randn(size(tx));
    rx = conv(rx, h);  % matched filter
    
    % Feature dimensions
    if neighbors_only
        n_win = 1 + 2 * num_nb;
    else
        n_win = win_len;
    end
    n_nb = 0;
    if ~neighbors_only && num_nb > 0
        n_nb = 2 * num_nb;
    end
    n_features = n_win + n_nb + num_fb;
    
    X = zeros(N, n_features);
    Y = zeros(N, 1);
    
    df_history = zeros(num_fb, 1);
    start_idx = max([num_fb + 1, num_nb + 1]);
    
    for i = start_idx:N
        center = (i - 1) * step + 1 + delay;
        
        if center - half_win < 1 || center + half_win > length(rx)
            continue;
        end
        if num_nb > 0 && (center - num_nb*step < 1 || center + num_nb*step > length(rx))
            continue;
        end
        
        % Window or Neighbors features
        if neighbors_only
            win_feat = zeros(1, 1 + 2*num_nb);
            win_feat(1) = rx(center);
            for k = 1:num_nb
                win_feat(1 + k) = rx(center - k*step);
                win_feat(1 + num_nb + k) = rx(center + k*step);
            end
        else
            if win_len == 1
                win_feat = rx(center);
            else
                win_feat = rx(center - half_win : center + half_win)';
            end
        end
        
        % Neighbor samples (for Full model)
        if ~neighbors_only && num_nb > 0
            nb_feat = zeros(1, 2 * num_nb);
            for k = 1:num_nb
                nb_feat(k) = rx(center - k * step);
                nb_feat(num_nb + k) = rx(center + k * step);
            end
        else
            nb_feat = [];
        end
        
        % Decision feedback
        if num_fb > 0
            df_feat = df_history';
        else
            df_feat = [];
        end
        
        X(i, :) = [win_feat, nb_feat, df_feat];
        Y(i) = bits(i);
        
        % Update DF history
        if num_fb > 0
            if rand() < tf_ratio
                new_decision = symbols(i);
            else
                new_decision = 0;
            end
            df_history = [new_decision; df_history(1:end-1)];
        end
    end
    
    valid = any(X, 2);
    X = X(valid, :);
    Y = Y(valid);
    fprintf('  Data: %d samples, %d features\n', size(X, 1), size(X, 2));
end

function net = train_model(X, Y)
    Y_cat = categorical(Y, [0 1]);
    
    n_in = size(X, 2);
    n_h1 = min(128, max(32, n_in * 2));
    n_h2 = min(64, max(16, n_in));
    
    layers = [
        featureInputLayer(n_in, 'Normalization', 'zscore')
        fullyConnectedLayer(n_h1)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(n_h2)
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];
    
    opts = trainingOptions('adam', ...
        'MaxEpochs', 25, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);
    
    net = trainNetwork(X, Y_cat, layers, opts);
    
    pred = classify(net, X);
    acc = mean(pred == Y_cat) * 100;
    fprintf('  Accuracy: %.2f%%\n', acc);
end
