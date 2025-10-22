clear; clc; close all;

tau = .8;
window_len = 31;
num_feedback = 4;
SNR_train_dB = 8;
N_symbols = 80000;
teacher_forcing_ratio = 0.5;

input_size = window_len + num_feedback;

fprintf('Training BPSK DF Equalizer with tau=%.1f at SNR=%d dB (TF=%.1f)\n', tau, SNR_train_dB, teacher_forcing_ratio);

[X_train, Y_train] = generate_bpsk_df_data(N_symbols, tau, SNR_train_dB, window_len, num_feedback, teacher_forcing_ratio);

fprintf('Generated %d training samples\n', size(X_train, 1));
fprintf('  Class 0 (bit=0): %d samples\n', sum(Y_train==0));
fprintf('  Class 1 (bit=1): %d samples\n', sum(Y_train==1));

Y_train_cat = categorical(Y_train, [0 1], {'0', '1'});

layers = [
    featureInputLayer(input_size, 'Normalization', 'zscore')
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

fprintf('Training network...\n');

cv = cvpartition(size(X_train, 1), 'HoldOut', 0.15);
X_tr = X_train(cv.training, :);
Y_tr = Y_train_cat(cv.training);
X_val = X_train(cv.test, :);
Y_val = Y_train_cat(cv.test);

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.002, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(X_tr, Y_tr, layers, options);

val_pred = predict(net, X_val);
[~, val_pred_class] = max(val_pred, [], 2);
val_accuracy = mean(val_pred_class == double(Y_val)) * 100;
fprintf('Final validation accuracy: %.2f%%\n', val_accuracy);

if ~exist('mat/bpsk_simple', 'dir'), mkdir('mat/bpsk_simple'); end
fname = sprintf('mat/bpsk_simple/bpsk_df_tau%02d.mat', tau*10);
save(fname, 'net', 'tau', 'window_len', 'num_feedback', 'input_size');
fprintf('Model saved: %s\n', fname);

function [X, Y] = generate_bpsk_df_data(N, tau, SNR_dB, win_len, num_fb, tf_ratio)
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
    
    for i = (num_fb+1):N
        idx = (i-1)*round(sps*tau) + 1 + delay;
        
        if idx > half_win && idx + half_win <= length(rx)
            win = rx(idx-half_win:idx+half_win);
            features = [real(win(:))', decision_history'];
            
            X(i, :) = features;
            Y(i) = bits(i);
            
            % if rand() < tf_ratio
            %     decision_history = [bits(i); decision_history(1:end-1)];
            % else
            %     pred_bit = round(rand());
            %     decision_history = [pred_bit; decision_history(1:end-1)];
            % end
        end
    end
    
    valid = any(X, 2);
    X = X(valid, :);
    Y = Y(valid);
end