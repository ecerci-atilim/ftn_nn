% =========================================================================
% FINAL TRAINING SCRIPT: The Decision Feedback CNN (DF-CNN)
% =========================================================================
% GOAL:
% To train a high-performance, specialist detector using a 1D-CNN
% architecture, which is fundamentally better suited for signal processing.
% Uses first-principles noise generation for maximum accuracy.
% =========================================================================

clear, clc, close all

%% --- PHASE 1: SETUP ---
fprintf('--- PHASE 1: Setting up the DF-CNN Training ---\n');

% --- Select the tau value for this specialist model ---
tau = 0.7;

% --- Network Hyperparameters ---
window_len = 31;
num_feedback_taps = 4;
input_len = window_len + num_feedback_taps; % 31 signal + 4 history

% --- Define the SUPERIOR CNN Architecture ---
layers = [
    featureInputLayer(input_len, 'Name', 'Input (Signal+History)')
    
    % The CNN needs the data reshaped to have a "channel" dimension
    functionLayer(@(X) reshape(X, [], 1, 1, size(X,1)), 'Formattable', true, 'Name', 'reshape')
    
    % 1D Convolutional Layers to act as learned filters
    convolution2dLayer([5 1], 32, 'Padding', 'causal', 'Name', 'conv1') % We use 2D conv to simulate 1D
    reluLayer('Name', 'relu1')
    batchNormalizationLayer('Name', 'bn1')
    
    convolution2dLayer([7 1], 64, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    
    % Aggregate features before the final decision
    globalAveragePooling2dLayer('Name', 'gap')
    
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];
% NOTE: We use 2D layers on a [Heightx1] input to simulate 1D convolution,
% which is a standard and robust practice in MATLAB.

%% --- PHASE 2: GENERATE HIGH-QUALITY TRAINING DATA ---
fprintf('--- PHASE 2: Generating training data for tau = %.1f ---\n', tau);
N_train = 60000;
SNR_train_dB = 20; % Train on a very clean signal to learn the ISI structure perfectly

[x_train, y_train] = generate_df_cnn_data(N_train, tau, SNR_train_dB, window_len, num_feedback_taps);
Y_cat_train = categorical(y_train);
fprintf('Generated %d training samples.\n', size(x_train, 1));

%% --- PHASE 3: TRAIN THE CNN DETECTOR ---
fprintf('--- PHASE 3: Training the CNN...\n');
cv = cvpartition(size(x_train, 1), 'HoldOut', 0.1);
X_train_final = x_train(cv.training, :);
Y_train_final = Y_cat_train(cv.training, :);
X_val = x_train(cv.test, :);
Y_val = Y_cat_train(cv.test, :);

options = trainingOptions('adam', 'MaxEpochs', 12, 'MiniBatchSize', 128, ...
    'ValidationData', {X_val, Y_val}, 'ValidationFrequency', 100, ...
    'InitialLearnRate', 0.001, 'Plots', 'training-progress', 'Verbose', false);

df_cnn_model = trainNetwork(X_train_final, Y_train_final, layers, options);

%% --- PHASE 4: SAVE THE SPECIALIST MODEL ---
fprintf('--- PHASE 4: Saving the final model...\n');
if ~exist('mat', 'dir'), mkdir('mat'); end
fname = sprintf('mat/df_cnn_detector_tau_%02d.mat', tau*10);
save(fname, 'df_cnn_model');
fprintf('Model saved successfully to "%s".\n', fname);

%% --- HELPER FUNCTION FOR DF-CNN TRAINING DATA ---
function [x,y] = generate_df_cnn_data(N, tau, SNR_dB, win_len, num_taps)
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    
    % --- MANUAL NOISE GENERATION ---
    % 1. Normalize transmitted signal power to 1
    pwr = mean(txSignal.^2);
    txSignal = txSignal / sqrt(pwr);
    % 2. Convert SNR from dB to linear
    snr_lin = 10^(SNR_dB/10);
    % 3. Calculate noise variance for unit power signal
    noise_variance = 1 / (2*snr_lin);
    % 4. Generate noise and add it
    noise = sqrt(noise_variance) * randn(size(txSignal));
    rxSignal = txSignal + noise;
    
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    x = zeros(N, win_len + num_taps);
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    for i = (num_taps + 1):N
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            win = rxMF(loc-half_win:loc+half_win).';
            hist = symbols(i-1:-1:i-num_taps).';
            x(i, :) = [win, hist];
            y(i) = bits(i);
        end
    end
    valid = any(x,2);
    x = x(valid,:);
    y = y(valid,:);
end