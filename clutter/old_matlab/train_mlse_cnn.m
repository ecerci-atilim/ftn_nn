
clear, clc, close all

%% SETUP
fprintf('--- PHASE 1: Setting up the DF-CNN Training ---\n');
tau = 0.5; % Specialist for a challenging tau
window_len = 31;
num_feedback_taps = 4;
input_len = window_len + num_feedback_taps;

%% DEFINE THE SUPERIOR CNN ARCHITECTURE
fprintf('--- PHASE 2: Defining the DF-CNN Architecture ---\n');
layers = [
    featureInputLayer(input_len, 'Name', 'input_vector', 'Normalization', 'zscore')
    % Custom layer to split the input into signal and history branches
    functionLayer(@splitInputs, 'Formattable', true, 'OutputNames', {'signal', 'history'}, 'Name', 'splitter')
];

lgraph = layerGraph(layers);

% CNN path for the signal
signal_branch = [
    % Reshape the signal part for convolution with proper formatting
    functionLayer(@(X) reshapeForCNNWithFormat(X, window_len), 'Formattable', true, 'Name', 'reshape_cnn')
    convolution2dLayer([1 7], 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    convolution2dLayer([1 5], 64, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    globalAveragePooling2dLayer('Name', 'gap1')
    flattenLayer('Name', 'flatten_cnn')
];
lgraph = addLayers(lgraph, signal_branch);
lgraph = connectLayers(lgraph, 'splitter/signal', 'reshape_cnn');

% FNN path for the history
history_branch = [
    fullyConnectedLayer(16, 'Name', 'fc_history')
    reluLayer('Name', 'relu_history')
];
lgraph = addLayers(lgraph, history_branch);
lgraph = connectLayers(lgraph, 'splitter/history', 'fc_history');

% Merge and Final Classifier
main_trunk = [
    concatenationLayer(1, 2, 'Name', 'concat')
    fullyConnectedLayer(64, 'Name', 'fc_merge')
    reluLayer('Name', 'relu_merge')
    dropoutLayer(0.3, 'Name', 'dropout1')
    fullyConnectedLayer(2, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];
lgraph = addLayers(lgraph, main_trunk);
lgraph = connectLayers(lgraph, 'flatten_cnn', 'concat/in1');
lgraph = connectLayers(lgraph, 'relu_history', 'concat/in2');

analyzeNetwork(lgraph);

%% GENERATE TRAINING DATA
fprintf('\n--- PHASE 3: Generating training data for tau = %.1f ---\n', tau);
N_train = 80000; % More data for the more powerful model
SNR_train_dB = 20;
[x_train, y_train] = generate_df_data(N_train, tau, SNR_train_dB, window_len, num_feedback_taps);
Y_cat_train = categorical(y_train);

%% TRAIN THE CNN DETECTOR
fprintf('\n--- PHASE 4: Training the CNN...\n');
cv = cvpartition(size(x_train, 1), 'HoldOut', 0.1);
X_train_final = x_train(cv.training, :);
Y_train_final = Y_cat_train(cv.training, :);
X_val = x_train(cv.test, :);
Y_val = Y_cat_train(cv.test, :);
options = trainingOptions('adam', 'MaxEpochs', 12, 'MiniBatchSize', 256, ...
    'ValidationData', {X_val, Y_val}, 'ValidationFrequency', 200, ...
    'InitialLearnRate', 0.001, 'Plots', 'training-progress', 'Verbose', false);
df_cnn_model = trainNetwork(X_train_final, Y_train_final, lgraph, options);

%% --- PHASE 5: SAVE THE MODEL ---
fprintf('\n--- PHASE 5: Saving the final model...\n');
if ~exist('mat', 'dir'), mkdir('mat'); end
fname = sprintf('mat/df_cnn_detector_tau_%02d.mat', tau*10);
save(fname, 'df_cnn_model');
fprintf('Model saved successfully to "%s".\n', fname);

%% --- HELPER FUNCTIONS ---
function [signal, history] = splitInputs(input)
    % This function splits the flat input vector into its two parts
    % Input comes as a feature vector from featureInputLayer
    % During training: input is (inputSize × batchSize)
    
    if isa(input, 'dlarray')
        % Extract underlying data if it's a dlarray
        inputData = extractdata(input);
        signal = dlarray(inputData(1:31, :), 'CB');
        history = dlarray(inputData(32:35, :), 'CB');
    else
        % Handle regular arrays
        signal = input(1:31, :);
        history = input(32:35, :);
    end
end

function output = reshapeForCNNWithFormat(X, window_len)
    % Reshape signal data for 2D convolution with proper formatting
    % Input: (window_len × batchSize) with format 'CB'
    % Output: (1 × window_len × 1 × batchSize) with format 'SSCB'
    
    if isa(X, 'dlarray')
        % Extract data and get dimensions
        data = extractdata(X);
        batchSize = size(data, 2);
        
        % Reshape to 4D for 2D convolution
        reshaped = reshape(data, 1, window_len, 1, batchSize);
        
        % Create properly formatted dlarray with spatial dimensions
        output = dlarray(reshaped, 'SSCB');
    else
        % Handle regular arrays during initialization
        batchSize = size(X, 2);
        output = reshape(X, 1, window_len, 1, batchSize);
    end
end

function [x,y] = generate_df_data(N, tau, SNR_dB, win_len, num_taps)
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(txSignal.^2); txSignal = txSignal / sqrt(pwr);
    snr_lin = 10^(SNR_dB/10); noise_var = 1 / (2*snr_lin);
    noise = sqrt(noise_var) * randn(size(txSignal));
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