clear; clc; close all;

tau = 0.7;
window_len = 31;
num_feedback = 4;
SNR_train_dB = 8;
N_symbols = 80000;
teacher_forcing_ratio = 0.5;

fprintf('=== Training Four Models for Comparison ===\n\n');

fprintf('Model 1: Window ONLY (31 samples, no DF)\n');
[X1, Y1] = generate_data(N_symbols, tau, SNR_train_dB, window_len, 0, 0);
net1 = train_model(X1, Y1, 'Window Only');

fprintf('\nModel 2: Window + DF (31 samples + 4 DF taps)\n');
[X2, Y2] = generate_data(N_symbols, tau, SNR_train_dB, window_len, num_feedback, teacher_forcing_ratio);
net2 = train_model(X2, Y2, 'Window + DF');

fprintf('\nModel 3: Single Sample + DF (1 sample + 4 DF taps) - Professor''s approach\n');
[X3, Y3] = generate_data(N_symbols, tau, SNR_train_dB, 1, num_feedback, teacher_forcing_ratio);
net3 = train_model(X3, Y3, 'Single + DF');

fprintf('\nModel 4: Single Sample ONLY (1 sample, no DF) - Absolute minimum\n');
[X4, Y4] = generate_data(N_symbols, tau, SNR_train_dB, 1, 0, 0);
net4 = train_model(X4, Y4, 'Single Only');

if ~exist('mat/comparison', 'dir'), mkdir('mat/comparison'); end
fname = sprintf('mat/comparison/comparison_tau%02d.mat', tau*10);
save(fname, 'net1', 'net2', 'net3', 'net4', 'tau', 'window_len', 'num_feedback');
fprintf('\n=== All models saved: %s ===\n', fname);

function [X, Y] = generate_data(N, tau, SNR_dB, win_len, num_fb, tf_ratio)
    sps = 10;
    span = 6;
    h = rcosdesign(0.3, span, sps, 'sqrt');
    h = h / norm(h);
    
    bits = randi([0 1], N, 1);
    symbols = 2*bits - 1;
    
    tx = upsample(symbols, round(sps*tau));
    tx = conv(tx, h);
    
    EbN0_linear = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_linear);
    noise = sqrt(noise_var) * randn(size(tx));
    rx = tx + noise;
    
    rx = conv(rx, h);
    
    delay = span * sps + 1;
    half_win = floor(win_len/2);
    
    X = zeros(N, win_len + num_fb);
    Y = zeros(N, 1);
    
    decision_history = zeros(num_fb, 1);
    
    start_idx = max(num_fb+1, 1);
    
    for i = start_idx:N
        idx = (i-1)*round(sps*tau) + 1 + delay;
        
        if idx > half_win && idx + half_win <= length(rx)
            if win_len == 1
                win_features = real(rx(idx));
            else
                win = rx(idx-half_win:idx+half_win);
                win_features = real(win(:))';
            end
            
            if num_fb > 0
                features = [win_features, decision_history'];
            else
                features = win_features;
            end
            
            X(i, :) = features;
            Y(i) = bits(i);
            
            if num_fb > 0
                if rand() < tf_ratio
                    decision_history = [bits(i); decision_history(1:end-1)];
                else
                    pred_bit = round(rand());
                    decision_history = [pred_bit; decision_history(1:end-1)];
                end
            end
        end
    end
    
    valid = any(X, 2);
    X = X(valid, :);
    Y = Y(valid);
end

function net = train_model(X, Y, model_name)
    fprintf('  Training %s (%d features)...\n', model_name, size(X, 2));
    
    Y_cat = categorical(Y, [0 1], {'0', '1'});
    
    layers = [
        featureInputLayer(size(X, 2), 'Normalization', 'zscore')
        fullyConnectedLayer(64)
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 512, ...
        'InitialLearnRate', 0.002, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net = trainNetwork(X, Y_cat, layers, options);
    
    pred = predict(net, X);
    [~, pred_class] = max(pred, [], 2);
    acc = mean(pred_class == double(Y_cat)) * 100;
    fprintf('  Training accuracy: %.2f%%\n', acc);
end