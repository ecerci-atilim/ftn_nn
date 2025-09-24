% =========================================================================
% QUASI-STATIC CHANNEL DF-CNN: Training Script
% =========================================================================
% Trains DF-CNN for quasi-static fading channels with multipath propagation
% Channel remains constant over coherence blocks but varies between blocks
% =========================================================================

clear, clc, close all

%% SETUP - Configure Modulation and Channel
fprintf('--- PHASE 1: Setting up Quasi-Static Channel DF-CNN Training ---\n');
modulation_type = 'psk';    % 'psk' or 'qam'
modulation_order = 4;       % 2, 4, 8, 16, 64, etc.
phase_offset = pi/4;        % Phase rotation in radians
tau = 0.8;                  % Timing parameter

% Quasi-static channel parameters
coherence_length = 100;     % Symbols per coherence block
num_taps_channel = 3;       % Number of multipath taps
max_delay_spread = 2;       % Maximum delay spread in symbol periods
fading_type = 'rayleigh';   % 'rayleigh', 'rician', or 'nakagami'

% Network parameters
window_len = 31;
num_feedback_taps = 4;
input_len = window_len * 2 + num_feedback_taps + num_taps_channel;  % Include CSI

k = log2(modulation_order);
fprintf('Configuration: %d-%s, phase=%.3f rad, tau=%.1f\n', modulation_order, upper(modulation_type), phase_offset, tau);
fprintf('Channel: %d-tap %s fading, coherence=%d symbols\n', num_taps_channel, fading_type, coherence_length);

%% DEFINE THE QUASI-STATIC CHANNEL DF-CNN ARCHITECTURE
fprintf('--- PHASE 2: Defining Quasi-Static Channel DF-CNN Architecture ---\n');

lgraph = layerGraph();

% Single combined input (signal + feedback + CSI)
input_layer = featureInputLayer(input_len, 'Name', 'combined_input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, input_layer);

% Enhanced deep network for multipath channels
deep_layers = [
    fullyConnectedLayer(512, 'Name', 'fc1')  % Larger capacity for multipath
    reluLayer('Name', 'relu1')
    batchNormalizationLayer('Name', 'bn1')
    
    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    batchNormalizationLayer('Name', 'bn2')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(128, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    batchNormalizationLayer('Name', 'bn3')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    fullyConnectedLayer(64, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.4, 'Name', 'dropout3')
    
    fullyConnectedLayer(modulation_order, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];
lgraph = addLayers(lgraph, deep_layers);

% Simple connection
lgraph = connectLayers(lgraph, 'combined_input', 'fc1');

fprintf('Quasi-Static DF-CNN architecture created for %d-class classification\n', modulation_order);

%% GENERATE TRAINING DATA
fprintf('\n--- PHASE 3: Generating quasi-static channel training data ---\n');
N_train = 150000;  % More data needed for channel variations
SNR_train_dB = 15;  % Lower SNR due to fading
[x_train, y_train] = generate_quasi_static_data(N_train, tau, SNR_train_dB, window_len, num_feedback_taps, ...
    modulation_type, modulation_order, phase_offset, coherence_length, num_taps_channel, max_delay_spread, fading_type);
Y_cat_train = categorical(y_train);

fprintf('Generated %d training samples with channel variations\n', size(x_train, 1));
for i = 0:modulation_order-1
    fprintf('  Symbol %d: %d samples\n', i, sum(y_train==i));
end

%% TRAIN THE QUASI-STATIC DF-CNN
fprintf('\n--- PHASE 4: Training the Quasi-Static DF-CNN...\n');
cv = cvpartition(size(x_train, 1), 'HoldOut', 0.15);
X_train_final = x_train(cv.training, :);
Y_train_final = Y_cat_train(cv.training, :);
X_val = x_train(cv.test, :);
Y_val = Y_cat_train(cv.test, :);

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...  % More epochs for complex channel
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 0.0005, ...  % Lower LR for stability
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 300, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ValidationPatience', 10);

quasi_static_model = trainNetwork(X_train_final, Y_train_final, lgraph, options);

%% SAVE THE MODEL
fprintf('\n--- PHASE 5: Saving the Quasi-Static DF-CNN model...\n');
if ~exist('mat/quasi_static', 'dir'), mkdir('mat/quasi_static'); end
phase_str = strrep(sprintf('%.3f', phase_offset), '.', 'p');
fname = sprintf('mat/quasi_static/qs_df_%s%d_tau%02d_phase%s_taps%d.mat', ...
    modulation_type, modulation_order, tau*10, phase_str, num_taps_channel);
save(fname, 'quasi_static_model', 'modulation_type', 'modulation_order', 'phase_offset', ...
     'tau', 'window_len', 'num_feedback_taps', 'coherence_length', 'num_taps_channel', ...
     'max_delay_spread', 'fading_type');
fprintf('Model saved to: %s\n', fname);

%% QUICK PERFORMANCE TEST
fprintf('\n--- PHASE 6: Quick performance validation...\n');
y_pred_probs = predict(quasi_static_model, X_val);
[~, y_pred_idx] = max(y_pred_probs, [], 2);
y_pred = y_pred_idx - 1;

accuracy = mean(y_pred == (double(Y_val) - 1)) * 100;
ser = mean(y_pred ~= (double(Y_val) - 1)) * 100;

fprintf('Validation Results:\n');
fprintf('  Symbol Accuracy: %.2f%%\n', accuracy);
fprintf('  Symbol Error Rate: %.4f%%\n', ser);

if ser < 5.0
    fprintf('SUCCESS: Quasi-Static DF-CNN trained successfully!\n');
elseif ser < 15.0
    fprintf('GOOD: Promising results for fading channel\n');
else
    fprintf('WARNING: May need more training or different architecture\n');
end

%% HELPER FUNCTION - QUASI-STATIC CHANNEL DATA GENERATION
function [x, y] = generate_quasi_static_data(N, tau, SNR_dB, win_len, num_taps, mod_type, M, phase, coh_len, ch_taps, max_delay, fading_type)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Generate symbol stream
    symbol_indices = randi([0 M-1], N, 1);
    symbols = constellation(symbol_indices + 1);
    
    % Determine if modulation is real
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    % Pulse shaping
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(abs(txSignal).^2); 
    txSignal = txSignal / sqrt(pwr);
    
    % Generate quasi-static channel realizations
    num_blocks = ceil(N / coh_len);
    channel_responses = generate_channel_realizations(num_blocks, ch_taps, max_delay, fading_type);
    
    % Apply channel and noise
    rxSignal = apply_quasi_static_channel(txSignal, channel_responses, coh_len, sps, tau, SNR_dB, k, is_real_modulation);
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    
    % Extract features with CSI
    x = zeros(N, win_len*2 + num_taps + ch_taps);  % Include CSI
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    
    for i = (num_taps + 1):N
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            % Signal window
            win_complex = rxMF(loc-half_win:loc+half_win);
            
            if is_real_modulation
                win_features = [real(win_complex(:))', real(win_complex(:))'];
            else
                win_features = [real(win_complex(:))', imag(win_complex(:))'];
            end
            
            % Decision feedback
            past_symbols = symbol_indices(i-1:-1:i-num_taps);
            
            % Channel state information
            block_idx = ceil(i / coh_len);
            if block_idx <= size(channel_responses, 1)
                csi = channel_responses(block_idx, :);
            else
                csi = zeros(1, ch_taps);
            end
            
            % Combine all features
            x(i, :) = [win_features, past_symbols', csi];
            y(i) = symbol_indices(i);
        end
    end
    
    valid = any(x, 2);
    x = x(valid, :);
    y = y(valid, :);
end

function channel_responses = generate_channel_realizations(num_blocks, ch_taps, max_delay, fading_type)
    % Generate channel impulse responses for each coherence block
    channel_responses = zeros(num_blocks, ch_taps);
    
    % Delay profile (exponential decay)
    delays = linspace(0, max_delay, ch_taps);
    power_profile = exp(-delays / (max_delay/3));  % Exponential decay
    power_profile = power_profile / sum(power_profile);  % Normalize
    
    for block = 1:num_blocks
        switch lower(fading_type)
            case 'rayleigh'
                % Rayleigh fading coefficients
                h = sqrt(power_profile/2) .* (randn(1, ch_taps) + 1j*randn(1, ch_taps));
            case 'rician'
                % Rician fading (K-factor = 3 dB)
                K = 10^(3/10);
                los_component = sqrt(K/(K+1)) * sqrt(power_profile);
                scattered_component = sqrt(power_profile/(2*(K+1))) .* (randn(1, ch_taps) + 1j*randn(1, ch_taps));
                h = los_component + scattered_component;
            case 'nakagami'
                % Nakagami fading (m = 1.5)
                m = 1.5;
                h = sqrt(power_profile) .* sqrt(gamrnd(m, 1/m, 1, ch_taps)) .* exp(1j*2*pi*rand(1, ch_taps));
            otherwise
                error('Unknown fading type: %s', fading_type);
        end
        channel_responses(block, :) = h;
    end
end

function rxSignal = apply_quasi_static_channel(txSignal, channel_responses, coh_len, sps, tau, SNR_dB, k, is_real_modulation)
    rxSignal = zeros(size(txSignal));
    
    % Apply channel block by block
    samples_per_block = coh_len * round(sps * tau);
    
    for block = 1:size(channel_responses, 1)
        start_idx = (block-1) * samples_per_block + 1;
        end_idx = min(start_idx + samples_per_block - 1, length(txSignal));
        
        if start_idx <= length(txSignal)
            tx_block = txSignal(start_idx:end_idx);
            h = channel_responses(block, :);
            
            % Convolve with channel
            rx_block = conv(tx_block, h, 'same');
            rxSignal(start_idx:end_idx) = rx_block;
        end
    end
    
    % Add noise with proper Eb/N0 scaling
    snr_eb_n0 = 10^(SNR_dB/10);
    snr_es_n0 = k * snr_eb_n0;
    signal_power = mean(abs(rxSignal).^2);
    noise_power = signal_power / snr_es_n0;
    
    if is_real_modulation
        noise = sqrt(noise_power) * randn(size(rxSignal));
    else
        noise = sqrt(noise_power/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
    end
    
    rxSignal = rxSignal + noise;
end

function constellation = generate_constellation(M, type, phase)
    if strcmpi(type, 'psk')
        p = 0:M-1;
        constellation = exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam')
        k = log2(M);
        if mod(k,2) ~= 0
            error('QAM order must be a power of 4');
        end
        n = sqrt(M);
        vals = -(n-1):2:(n-1);
        [X,Y] = meshgrid(vals, vals);
        constellation = (X + 1j*Y) * exp(1j*phase);
        constellation = constellation(:);
    else
        error('Unknown modulation type. Use "psk" or "qam".');
    end
    
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end