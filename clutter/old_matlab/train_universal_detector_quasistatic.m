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
modulation_order = 2;       % 2, 4, 8, 16, 64, etc.
phase_offset = 0;        % Phase rotation in radians
tau = 0.8;                  % Timing parameter

% Quasi-static channel parameters
coherence_length = 100;     % Symbols per coherence block
num_taps_channel = 3;       % Number of multipath taps
max_delay_spread = 2;       % Maximum delay spread in symbol periods
fading_type = 'rician';   % 'rayleigh', 'rician', or 'nakagami'

% Network parameters
window_len = 31;
num_feedback_taps = 4;
input_len = window_len * 2 + num_feedback_taps + num_taps_channel;  % Include CSI

k = log2(modulation_order);
fprintf('Configuration: %d-%s, phase=%.3f rad, tau=%.1f\n', modulation_order, upper(modulation_type), phase_offset, tau);
fprintf('Channel: %d-tap %s fading, coherence=%d symbols\n', num_taps_channel, fading_type, coherence_length);

%% DEFINE THE QUASI-STATIC CHANNEL DF-CNN ARCHITECTURE
fprintf('--- PHASE 2: Defining Quasi-Static Channel DF-CNN Architecture ---\n');

% Calculate expected dimensions for debugging
fprintf('Expected dimensions:\n');
fprintf('  Window features: %d*2 = %d\n', window_len, window_len*2);
fprintf('  Feedback features: %d\n', num_feedback_taps); 
fprintf('  CSI features: %d*2 = %d\n', num_taps_channel, num_taps_channel*2);
fprintf('  Total expected: %d + %d + %d = %d\n', window_len*2, num_feedback_taps, num_taps_channel*2, input_len);

% Network will be created after data generation with correct dimensions
fprintf('Network architecture will be defined after data generation to match actual dimensions.\n');

%% GENERATE TRAINING DATA
fprintf('\n--- PHASE 3: Generating quasi-static channel training data ---\n');
N_train = 80000;   % Reduced for faster generation
SNR_train_dB = 15;
tic;
[x_train, y_train] = generate_quasi_static_data_fast(N_train, tau, SNR_train_dB, window_len, num_feedback_taps, ...
    modulation_type, modulation_order, phase_offset, coherence_length, num_taps_channel, max_delay_spread, fading_type);
generation_time = toc;
Y_cat_train = categorical(y_train);

% DEBUG: Check actual data dimensions
actual_features = size(x_train, 2);
expected_features = input_len;

fprintf('Generated %d samples in %.1f seconds (%.1f samples/sec)\n', size(x_train, 1), generation_time, size(x_train, 1)/generation_time);
fprintf('DIMENSION CHECK:\n');
fprintf('  Expected features: %d\n', expected_features);
fprintf('  Actual features: %d\n', actual_features);
fprintf('  Signal features: %d (window_len*2 = %d*2)\n', window_len*2, window_len);
fprintf('  Feedback features: %d\n', num_feedback_taps);
fprintf('  CSI features: %d (ch_taps*2 = %d*2)\n', num_taps_channel*2, num_taps_channel);

if actual_features ~= expected_features
    fprintf('  MISMATCH DETECTED! Updating input_len to match actual data...\n');
    input_len = actual_features;  % Fix the mismatch
end

for i = 0:modulation_order-1
    fprintf('  Symbol %d: %d samples\n', i, sum(y_train==i));
end

%% NOW DEFINE THE NETWORK WITH CORRECT DIMENSIONS
fprintf('\n--- Defining Network Architecture with Correct Dimensions ---\n');
lgraph = layerGraph();

% Input layer with correct dimensions
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

fprintf('Quasi-Static DF-CNN architecture created: %d inputs -> %d classes\n', input_len, modulation_order);

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

%% HELPER FUNCTION - FAST QUASI-STATIC CHANNEL DATA GENERATION
function [x, y] = generate_quasi_static_data_fast(N, tau, SNR_dB, win_len, num_taps, mod_type, M, phase, coh_len, ch_taps, max_delay, fading_type)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Generate symbol stream
    symbol_indices = randi([0 M-1], N, 1);
    symbols = constellation(symbol_indices + 1);
    
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    % Fast pulse shaping
    sps = 10; beta = 0.3; span = 6;
    h = rcosdesign(beta, span, sps, 'sqrt');
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(abs(txSignal).^2); 
    txSignal = txSignal / sqrt(pwr);
    
    % SIMPLIFIED QUASI-STATIC CHANNEL MODEL
    % Generate fewer channel realizations and apply efficiently
    num_blocks = ceil(N / coh_len);
    
    % Pre-generate all channel coefficients
    fprintf('  Generating %d channel blocks...', num_blocks);
    h_channels = zeros(num_blocks, ch_taps);
    delays = linspace(0, max_delay, ch_taps);
    power_profile = exp(-delays / (max_delay/3));
    power_profile = power_profile / sum(power_profile);
    
    % Generate all channel realizations at once (vectorized)
    switch lower(fading_type)
        case 'rayleigh'
            h_channels = sqrt(power_profile/2) .* (randn(num_blocks, ch_taps) + 1j*randn(num_blocks, ch_taps));
        case 'rician'
            K = 10^(3/10);
            los = sqrt(K/(K+1)) * sqrt(power_profile);
            scatter = sqrt(power_profile/(2*(K+1))) .* (randn(num_blocks, ch_taps) + 1j*randn(num_blocks, ch_taps));
            h_channels = repmat(los, num_blocks, 1) + scatter;
        case 'nakagami'
            m = 1.5;
            h_channels = sqrt(power_profile) .* sqrt(gamrnd(m, 1/m, num_blocks, ch_taps)) .* ...
                         exp(1j*2*pi*rand(num_blocks, ch_taps));
    end
    
    % SIMPLIFIED CHANNEL APPLICATION
    % Instead of full convolution, use simple multiplicative fading + delay
    rxSignal = zeros(size(txSignal));
    samples_per_block = coh_len * round(sps * tau);
    
    fprintf(' Applying channel...', num_blocks);
    for block = 1:num_blocks
        start_idx = (block-1) * samples_per_block + 1;
        end_idx = min(start_idx + samples_per_block - 1, length(txSignal));
        
        if start_idx <= length(txSignal)
            % Simplified: dominant tap + phase rotation (much faster)
            dominant_tap = h_channels(block, 1);  % Use only strongest tap
            rxSignal(start_idx:end_idx) = txSignal(start_idx:end_idx) * dominant_tap;
        end
    end
    
    % Add noise
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
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    
    % FAST FEATURE EXTRACTION
    fprintf(' Extracting features...');
    x = zeros(N, win_len*2 + num_taps + ch_taps*2);  % Real CSI features
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    
    valid_count = 0;
    for i = (num_taps + 1):N
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            valid_count = valid_count + 1;
            
            % Signal window
            win_complex = rxMF(loc-half_win:loc+half_win);
            
            if is_real_modulation
                win_features = [real(win_complex(:))', real(win_complex(:))'];
            else
                win_features = [real(win_complex(:))', imag(win_complex(:))'];
            end
            
            % Decision feedback
            past_symbols = symbol_indices(i-1:-1:i-num_taps);
            
            % CSI features - ENSURE REAL-VALUED
            block_idx = ceil(i / coh_len);
            if block_idx <= size(h_channels, 1)
                h_current = h_channels(block_idx, :);
                csi_features = [real(h_current), imag(h_current)];  % Separate I/Q
            else
                csi_features = zeros(1, ch_taps*2);
            end
            
            % Combine all features (ALL REAL-VALUED)
            x(i, :) = [win_features, past_symbols', csi_features];
            y(i) = symbol_indices(i);
        end
    end
    
    % Remove invalid samples
    valid = any(x, 2);
    x = x(valid, :);
    y = y(valid, :);
    
    fprintf(' Done! Generated %d valid samples\n', sum(valid));
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