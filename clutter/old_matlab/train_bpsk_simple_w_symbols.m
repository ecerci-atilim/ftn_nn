% çalışıyor

clear; clc; close all;

tau = .5;
window_len = 31;
SNR_train_dB = 5;
N_symbols = 80000;
num_past_future_symbols = 3; % New parameter: number of past/future symbols to include

input_size = window_len + (2 * num_past_future_symbols); % Update input_size

fprintf('Training BPSK Equalizer (no DF) with tau=%.1f at SNR=%d dB\n', tau, SNR_train_dB);
fprintf('Including %d past and %d future symbols as features.\n', num_past_future_symbols, num_past_future_symbols);

[X_train, Y_train] = generate_bpsk_data(N_symbols, tau, SNR_train_dB, window_len, num_past_future_symbols);

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
fname = sprintf('mat/bpsk_simple/bpsk_tau%02d_extfeat.mat', tau*10); % Changed filename to indicate extended features
save(fname, 'net', 'tau', 'window_len', 'input_size', 'num_past_future_symbols'); % Save new parameter
fprintf('Model saved: %s\n', fname);

function [X, Y] = generate_bpsk_data(N, tau, SNR_dB, win_len, num_past_future_symbols)
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
    
    delay_system = span * sps; % System delay from filters
    symbol_spacing_samples = round(sps*tau); % Delay between symbols in samples
    
    half_win = floor(win_len/2);
    
    % The total input size will be win_len for the window samples
    % plus 2 * num_past_future_symbols for the additional symbol samples
    total_input_size = win_len + (2 * num_past_future_symbols);
    X = zeros(N, total_input_size);
    Y = zeros(N, 1);
    
    for i = 1:N
        % Current symbol's sampling point index in the received signal
        current_symbol_idx = (i-1)*symbol_spacing_samples + 1 + delay_system;
        
        % Calculate indices for past and future symbols
        past_symbol_indices = current_symbol_idx - (1:num_past_future_symbols) * symbol_spacing_samples;
        future_symbol_indices = current_symbol_idx + (1:num_past_future_symbols) * symbol_spacing_samples;
        
        all_relevant_indices = [past_symbol_indices, current_symbol_idx-half_win:current_symbol_idx+half_win, future_symbol_indices];
        
        % Check if all relevant indices are within the bounds of rx
        if min(all_relevant_indices) >= 1 && max(all_relevant_indices) <= length(rx)
            % Extract window samples around the current symbol
            win_samples = real(rx(current_symbol_idx-half_win:current_symbol_idx+half_win));
            
            % Extract past symbol samples
            past_samples = real(rx(past_symbol_indices));
            
            % Extract future symbol samples
            future_samples = real(rx(future_symbol_indices));
            
            % Concatenate all features
            features = [past_samples; win_samples; future_samples];
            
            X(i, :) = features(:)'; % Ensure it's a row vector
            Y(i) = bits(i);
        end
    end
    
    valid = any(X, 2); % Keep rows where features were successfully extracted
    X = X(valid, :);
    Y = Y(valid);
end