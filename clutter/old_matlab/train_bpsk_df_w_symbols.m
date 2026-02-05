clear; clc; close all;

tau = .5;
window_len = 31;
num_feedback = 4;
num_past_future_symbols = 3; % New parameter: number of past/future symbols to include
SNR_train_dB = 8;
N_symbols = 80000;
teacher_forcing_ratio = 0.5;

% Update input_size to include window_len, num_feedback, and 2 * num_past_future_symbols
input_size = window_len + num_feedback + (2 * num_past_future_symbols);

fprintf('Training BPSK DF Equalizer with tau=%.1f at SNR=%d dB (TF=%.1f)\n', tau, SNR_train_dB, teacher_forcing_ratio);
fprintf('Including %d feedback taps and %d past/future symbols as features.\n', num_feedback, num_past_future_symbols);

[X_train, Y_train] = generate_bpsk_df_data(N_symbols, tau, SNR_train_dB, window_len, num_feedback, num_past_future_symbols, teacher_forcing_ratio);

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
% Changed filename to indicate extended features with DF
fname = sprintf('mat/bpsk_simple/bpsk_df_tau%02d_extfeat.mat', tau*10);
% Save new parameter num_past_future_symbols
save(fname, 'net', 'tau', 'window_len', 'num_feedback', 'num_past_future_symbols', 'input_size');
fprintf('Model saved: %s\n', fname);

function [X, Y] = generate_bpsk_df_data(N, tau, SNR_dB, win_len, num_fb, num_past_future_symbols, tf_ratio)
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
    
    delay_system = span * sps + 1; % System delay from filters
    symbol_spacing_samples = round(sps*tau); % Delay between symbols in samples
    
    half_win = floor(win_len/2);
    
    % Total input size will be win_len for the window samples,
    % num_fb for feedback, and 2 * num_past_future_symbols for the additional symbol samples
    total_input_size = win_len + num_fb + (2 * num_past_future_symbols);
    X = zeros(N, total_input_size);
    Y = zeros(N, 1);
    
    % Initialize decision history for teacher forcing
    % Using true bits for the initial feedback to avoid issues at the start
    initial_decision_history = bits((num_fb+1)-num_fb : (num_fb+1)-1);
    decision_history = initial_decision_history; % This should contain bits, not symbols
    
    for i = (num_fb+1):N % Start after enough symbols for feedback and past samples
        current_symbol_idx = (i-1)*symbol_spacing_samples + 1 + delay_system;
        
        % Calculate indices for past and future symbols (excluding the current one)
        past_symbol_indices = current_symbol_idx - (1:num_past_future_symbols) * symbol_spacing_samples;
        future_symbol_indices = current_symbol_idx + (1:num_past_future_symbols) * symbol_spacing_samples;
        
        % Combine all relevant indices for boundary check
        all_relevant_indices = [past_symbol_indices, current_symbol_idx-half_win:current_symbol_idx+half_win, future_symbol_indices];
        
        % Check if all relevant indices are within the bounds of rx
        if min(all_relevant_indices) >= 1 && max(all_relevant_indices) <= length(rx)
            win_samples = real(rx(current_symbol_idx-half_win:current_symbol_idx+half_win));
            past_samples = real(rx(past_symbol_indices));
            future_samples = real(rx(future_symbol_indices));
            
            % Concatenate window samples, past/future samples, and feedback decisions
            % Ensure decision_history is a row vector for concatenation
            features = [past_samples; win_samples; future_samples; decision_history];
            
            X(i, :) = features;
            Y(i) = bits(i);
            
            % Teacher forcing logic (only for training data generation)
            if rand() < tf_ratio
                % Use true bit for feedback
                decision_history = [bits(i); decision_history(1:end-1)];
            else
                % Use a random bit for feedback (simulating an error)
                pred_bit = randi([0 1]);
                decision_history = [pred_bit; decision_history(1:end-1)];
            end
        end
    end
    
    % Remove unused rows (due to initialization and boundary checks)
    valid = any(X, 2); 
    X = X(valid, :);
    Y = Y(valid);
end