% =========================================================================
% FIXED UNIVERSAL DF-CNN: Using Your Proven Working Approach
% =========================================================================
% Based on your successful BPSK method, adapted for any PSK/QAM modulation
% =========================================================================

clear, clc, close all

%% SETUP - Configure Your Modulation Here
fprintf('--- PHASE 1: Setting up Universal DF-CNN Training ---\n');
modulation_type = 'qam';    % 'psk' or 'qam'
modulation_order = 16;      % 2, 4, 8, 16, 64, etc.
phase_offset = 0;        % Phase rotation in radians
tau = 1.0;                  % Your challenging tau
window_len = 31;
num_feedback_taps = 4;
input_len = window_len * 2 + num_feedback_taps;  % I/Q + feedback

k = log2(modulation_order);
fprintf('Configuration: %d-%s, phase=%.3f rad, tau=%.1f\n', modulation_order, upper(modulation_type), phase_offset, tau);

%% DEFINE THE SIMPLE WORKING ARCHITECTURE (NO CUSTOM SPLITTING)
fprintf('--- PHASE 2: Defining Simple Working DF-CNN Architecture ---\n');

lgraph = layerGraph();

% Single combined input (proven to work)
input_layer = featureInputLayer(input_len, 'Name', 'combined_input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, input_layer);

% Deep network (no splitting, just like your successful BPSK version)
deep_layers = [
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    batchNormalizationLayer('Name', 'bn1')
    
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    batchNormalizationLayer('Name', 'bn2')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.4, 'Name', 'dropout2')
    
    fullyConnectedLayer(modulation_order, 'Name', 'fc_final')  % Multi-class output
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];
lgraph = addLayers(lgraph, deep_layers);

% Simple connection (no custom layers)
lgraph = connectLayers(lgraph, 'combined_input', 'fc1');

fprintf('Simple DF-CNN architecture created for %d-class classification\n', modulation_order);

%% GENERATE TRAINING DATA
fprintf('\n--- PHASE 3: Generating training data ---\n');
N_train = 120000;
SNR_train_dB = 20;
[x_train, y_train] = generate_simple_df_data(N_train, tau, SNR_train_dB, window_len, num_feedback_taps, modulation_type, modulation_order, phase_offset);
Y_cat_train = categorical(y_train);

fprintf('Generated %d training samples\n', size(x_train, 1));
for i = 0:modulation_order-1
    fprintf('  Symbol %d: %d samples\n', i, sum(y_train==i));
end

%% TRAIN THE SIMPLE DF-CNN
fprintf('\n--- PHASE 4: Training the Simple DF-CNN...\n');
cv = cvpartition(size(x_train, 1), 'HoldOut', 0.15);
X_train_final = x_train(cv.training, :);
Y_train_final = Y_cat_train(cv.training, :);
X_val = x_train(cv.test, :);
Y_val = Y_cat_train(cv.test, :);

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 200, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ValidationPatience', 8);

universal_df_model = trainNetwork(X_train_final, Y_train_final, lgraph, options);

%% SAVE THE MODEL
fprintf('\n--- PHASE 5: Saving the Universal DF-CNN model...\n');
if ~exist('mat/universal_df', 'dir'), mkdir('mat/universal_df'); end
phase_str = strrep(sprintf('%.3f', phase_offset), '.', 'p');
fname = sprintf('mat/universal_df/simple_df_%s%d_tau%02d_phase%s.mat', ...
    modulation_type, modulation_order, tau*10, phase_str);
save(fname, 'universal_df_model', 'modulation_type', 'modulation_order', 'phase_offset', 'tau', 'window_len', 'num_feedback_taps');
fprintf('Model saved to: %s\n', fname);

%% QUICK PERFORMANCE TEST
fprintf('\n--- PHASE 6: Quick performance validation...\n');
y_pred_probs = predict(universal_df_model, X_val);  % Only one output for classification
[~, y_pred_idx] = max(y_pred_probs, [], 2);
y_pred = y_pred_idx - 1; % Convert to 0-based indexing

% DEBUG: Check data formats
fprintf('DEBUG INFO:\n');
fprintf('  y_pred_probs size: [%d, %d]\n', size(y_pred_probs));
fprintf('  y_pred range: %d to %d\n', min(y_pred), max(y_pred));
fprintf('  Y_val type: %s\n', class(Y_val));
fprintf('  Y_val range: %d to %d\n', min(double(Y_val)), max(double(Y_val)));
fprintf('  Y_val categories: %s\n', strjoin(string(categories(Y_val)), ', '));

% Check first few predictions
fprintf('  First 5 predictions vs true:\n');
for i = 1:min(5, length(y_pred))
    fprintf('    Sample %d: Pred=%d, True=%d\n', i, y_pred(i), double(Y_val(i)));
end

% CORRECTED COMPARISON: Y_val is categorical with 1-based indexing
accuracy = mean(y_pred == (double(Y_val) - 1)) * 100;  % Convert Y_val to 0-based
ser = mean(y_pred ~= (double(Y_val) - 1)) * 100;

fprintf('Validation Results:\n');
fprintf('  Symbol Accuracy: %.2f%%\n', accuracy);
fprintf('  Symbol Error Rate: %.4f%%\n', ser);

if ser < 1.0
    fprintf('SUCCESS: Simple DF-CNN achieved excellent performance!\n');
elseif ser < 5.0
    fprintf('GOOD: Simple DF-CNN shows promising results\n');
else
    fprintf('WARNING: May need more training or check data\n');
end

%% HELPER FUNCTION - CORRECTED DATA GENERATION WITH PROPER Eb/N0
function [x, y] = generate_simple_df_data(N, tau, SNR_dB, win_len, num_taps, mod_type, M, phase)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Generate random symbols
    symbol_indices = randi([0 M-1], N, 1);
    symbols = constellation(symbol_indices + 1);
    
    % Determine if modulation is real (BPSK only)
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    % Channel simulation
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(abs(txSignal).^2); txSignal = txSignal / sqrt(pwr);
    
    % CORRECTED NOISE GENERATION: Convert Eb/N0 to Es/N0
    snr_eb_n0 = 10^(SNR_dB/10);        % Eb/N0 in linear scale
    snr_es_n0 = k * snr_eb_n0;         % Convert to Es/N0
    signal_power = mean(abs(txSignal).^2);
    noise_power = signal_power / snr_es_n0;
    
    % Generate appropriate noise type
    if is_real_modulation  % BPSK with real symbols
        noise = sqrt(noise_power) * randn(size(txSignal));
    else  % Complex modulations (QPSK, QAM)
        noise = sqrt(noise_power/2) * (randn(size(txSignal)) + 1j*randn(size(txSignal)));
    end
    
    rxSignal = txSignal + noise;
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    
    % Extract features and labels
    x = zeros(N, win_len*2 + num_taps);  % I/Q + feedback
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    
    for i = (num_taps + 1):N
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            % Signal window - I/Q features for all modulations
            win_complex = rxMF(loc-half_win:loc+half_win);
            
            if is_real_modulation
                % For BPSK: use real samples twice to maintain dimension compatibility
                win_features = [real(win_complex(:))', real(win_complex(:))'];
            else
                % For complex modulations: proper I/Q features
                win_features = [real(win_complex(:))', imag(win_complex(:))'];
            end
            
            % Decision feedback from past symbols (as indices)
            past_symbols = symbol_indices(i-1:-1:i-num_taps);
            
            % Combine features
            x(i, :) = [win_features, past_symbols'];
            y(i) = symbol_indices(i);
        end
    end
    
    valid = any(x,2);
    x = x(valid,:);
    y = y(valid,:);
end

function constellation = generate_constellation(M, type, phase)
    if strcmpi(type, 'psk')
        p = 0:M-1;
        constellation = exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam')
        k = log2(M);
        if mod(k,2) ~= 0
            error('QAM order must be a power of 4 (4, 16, 64, ...)');
        end
        n = sqrt(M);
        vals = -(n-1):2:(n-1);
        [X,Y] = meshgrid(vals, vals);
        constellation = (X + 1j*Y) * exp(1j*phase);
        constellation = constellation(:);
    else
        error('Unknown modulation type. Use "psk" or "qam".');
    end
    
    % Normalize to unit average power
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end