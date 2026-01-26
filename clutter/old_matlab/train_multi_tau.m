%% FTN NN Training - Multiple Tau Values
% τ = 0.6, 0.7, 0.8 için model eğitimi
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

tau_values = [0.6, 0.7, 0.8];
window_len = 31;
num_feedback = 4;
num_neighbor = 3;
SNR_train_dB = 8;
N_symbols = 100000;
tf_ratio = 0.7;

for tau = tau_values
    fprintf('\n========================================\n');
    fprintf('=== Training for τ = %.1f ===\n', tau);
    fprintf('========================================\n');
    
    % Check if model exists
    fname = sprintf('mat/comparison/tau%02d.mat', tau*10);
    if exist(fname, 'file')
        fprintf('Model already exists: %s\n', fname);
        continue;
    end
    
    % Model 4: Neighbors+DF
    fprintf('Model 4: Neighbors+DF\n');
    [X4, Y4] = gen_data(N_symbols, tau, SNR_train_dB, 1, num_feedback, tf_ratio, num_neighbor, true);
    net4 = train_nn(X4, Y4);
    
    % Model 6: Window+DF
    fprintf('Model 6: Window+DF\n');
    [X6, Y6] = gen_data(N_symbols, tau, SNR_train_dB, window_len, num_feedback, tf_ratio, 0, false);
    net6 = train_nn(X6, Y6);
    
    % Save
    if ~exist('mat/comparison', 'dir'), mkdir('mat/comparison'); end
    save(fname, 'net4', 'net6', 'tau', 'window_len', 'num_feedback', 'num_neighbor');
    fprintf('Saved: %s\n', fname);
end

fprintf('\n=== All training complete ===\n');

%% Functions
function [X, Y] = gen_data(N, tau, SNR_dB, win_len, num_fb, tf_ratio, num_nb, neighbors_only)
    sps = 10; span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    step = round(sps * tau);
    delay = span * sps;
    half_win = floor(win_len / 2);
    
    bits = randi([0 1], N, 1);
    symbols = 2 * bits - 1;
    
    tx = conv(upsample(symbols, step), h);
    EbN0 = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0);
    rx = tx + sqrt(noise_var) * randn(size(tx));
    rx = conv(rx, h);
    rx = rx / std(rx);  % Normalization
    
    if neighbors_only
        n_win = 1 + 2 * num_nb;
    else
        n_win = win_len;
    end
    n_features = n_win + num_fb;
    
    X = zeros(N, n_features);
    Y = zeros(N, 1);
    df_hist = zeros(num_fb, 1);
    start_idx = max([num_fb + 1, num_nb + 1]);
    
    for i = start_idx:N
        ctr = (i - 1) * step + 1 + delay;
        if ctr - half_win < 1 || ctr + half_win > length(rx), continue; end
        if num_nb > 0 && (ctr - num_nb*step < 1 || ctr + num_nb*step > length(rx)), continue; end
        
        if neighbors_only
            feat = rx(ctr);
            for k = 1:num_nb, feat = [feat, rx(ctr - k*step)]; end
            for k = 1:num_nb, feat = [feat, rx(ctr + k*step)]; end
        else
            feat = rx(ctr - half_win : ctr + half_win)';
        end
        
        if num_fb > 0, feat = [feat, df_hist']; end
        
        X(i, :) = feat;
        Y(i) = bits(i);
        
        if num_fb > 0
            if rand() < tf_ratio
                df_hist = [symbols(i); df_hist(1:end-1)];
            else
                df_hist = [0; df_hist(1:end-1)];
            end
        end
    end
    
    valid = any(X, 2);
    X = X(valid, :);
    Y = Y(valid);
    fprintf('  Data: %d samples, %d features\n', size(X, 1), size(X, 2));
end

function net = train_nn(X, Y)
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
    acc = mean(classify(net, X) == Y_cat) * 100;
    fprintf('  Accuracy: %.2f%%\n', acc);
end