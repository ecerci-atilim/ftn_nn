% =========================================================================
% FINAL TRAINING SCRIPT: The "Oracle" MLSE-like Neural Network
% =========================================================================
% GOAL:
% To train a single, specialist network for a specific tau value using
% a three-part input: the signal window, perfect past symbols (feedback),
% and perfect future symbols (feedforward).
% =========================================================================

clear, clc, close all

%% --- PHASE 1: SETUP ---
fprintf('--- PHASE 1: Setting up the MLSE-like Network Training ---\n');

% --- Select the tau value for this specialist model ---
tau = 0.7; % << You can change this to 0.5, 0.8, etc.

% --- Network Hyperparameters ---
window_len = 31;
num_past_taps = 4;
num_future_taps = 2; % Looking 2 symbols into the future
% The final, complete input vector size:
input_len = window_len + num_past_taps + num_future_taps; % 31+4+2 = 37

% --- Define the Network Architecture ---
% This architecture is wide enough to handle the rich input data.
layers = [
    featureInputLayer(input_len, 'Name', 'Input (Past+Signal+Future)')
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

fprintf('Network designed for an input vector of size %d.\n', input_len);

%% --- PHASE 2: GENERATE "ORACLE" TRAINING DATA ---
fprintf('--- PHASE 2: Generating Oracle training data for tau = %.1f ---\n', tau);

% --- Simulation Parameters ---
N_train = 60000; % A large dataset for robust training
SNR_train = 15;  % High SNR for a clear signal

% --- Generate Data ---
% This helper function now generates the three-part input vectors.
[x_train, y_train] = generate_oracle_data(N_train, tau, SNR_train, window_len, num_past_taps, num_future_taps);
Y_cat_train = categorical(y_train);
fprintf('Generated %d high-quality training samples.\n', size(x_train, 1));


%% --- PHASE 3: TRAIN THE NETWORK ---
fprintf('--- PHASE 3: Training the Network...\n');

% Partition data for validation
cv = cvpartition(size(x_train, 1), 'HoldOut', 0.15);
X_train_final = x_train(cv.training, :);
Y_train_final = Y_cat_train(cv.training, :);
X_val = x_train(cv.test, :);
Y_val = Y_cat_train(cv.test, :);

% Training options
options = trainingOptions('adam', 'MaxEpochs', 15, 'MiniBatchSize', 128, ...
    'ValidationData', {X_val, Y_val}, 'ValidationFrequency', 100, ...
    'Verbose', true, 'Plots', 'training-progress');

% Train the model
mlse_model = trainNetwork(X_train_final, Y_train_final, layers, options);

%% --- PHASE 4: SAVE THE SPECIALIST MODEL ---
fprintf('--- PHASE 4: Saving the final model...\n');

% Create the 'mat' subfolder if it doesn't exist
if ~exist('mat', 'dir')
   mkdir('mat')
end
% Save the model into the 'mat' subfolder with a descriptive name
fname = sprintf('mat/mlse_nn_detector_tau_%02d.mat', tau*10);
save(fname, 'mlse_model');
fprintf('Model saved successfully to "%s".\n', fname);


%% --- HELPER FUNCTION FOR ORACLE TRAINING DATA ---
function [x,y] = generate_oracle_data(N, tau, SNR, win_len, num_past, num_future)
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_up = upsample(symbols, round(sps*tau));
    rxMF = awgn(conv(conv(tx_up, h), h), SNR, 'measured');
    delay = finddelay(tx_up, rxMF);
    x = zeros(N, win_len + num_past + num_future);
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    % Loop must stop early to ensure future symbols are available
    for i = (num_past + 1):(N - num_future)
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            win = rxMF(loc-half_win:loc+half_win).';
            % Use PERFECT ground-truth for past and future
            past = symbols(i-1:-1:i-num_past).';
            future = symbols(i+1:i+num_future).';
            x(i, :) = [win, past, future];
            y(i) = bits(i);
        end
    end
    valid = any(x,2);
    x = x(valid,:);
    y = y(valid,:);
end