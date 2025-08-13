% =========================================================================
% SCRIPT 1: Train the Final, High-Capacity Adaptive Detector
% =========================================================================
% GOAL:
% To create and save a SINGLE, robust neural network that can demodulate
% FTN signals over a wide range of acceleration factors (tau) by providing
% tau as a direct input. This version uses a larger network with dropout
% to ensure successful generalization across all trained tasks.
% =========================================================================

clear, clc, close all

%% PHASE 1: SETUP FOR ADAPTIVE TRAINING
fprintf('--- PHASE 1: Setting up for Adaptive Training ---\n');

% --- Define the contexts (tau values) the model must learn ---
tau_train_set = [0.5, 0.6, 0.7, 0.8, 0.9];
fprintf('Training a single model to handle tau values: %s\n', num2str(tau_train_set));

% --- System & Network Parameters ---
N_per_tau = 40000;          % Generate 40,000 examples for EACH tau
SNR_for_training = 15;      % Train on a clean, high-SNR signal
window_len = 31;            % The signal window size
num_feedback_taps = 4;      % The number of past decisions to use
adaptive_input_len = window_len + num_feedback_taps + 1; % 31(sig)+4(hist)+1(tau)=36

% --- Define the NEW, LARGER, and more ROBUST architecture ---
% This architecture has more capacity (more neurons) and better
% regularization (dropout) to handle multiple tasks without interference.
layers = [
    featureInputLayer(adaptive_input_len, 'Name', 'Input (Signal+History+Tau)')
    
    fullyConnectedLayer(256, 'Name', 'FC_1') % Increased size for more capacity
    reluLayer('Name', 'ReLU_1')
    dropoutLayer(0.3, 'Name', 'Dropout_1') % Add dropout to prevent overfitting
    
    fullyConnectedLayer(128, 'Name', 'FC_2') % Increased size
    reluLayer('Name', 'ReLU_2')
    dropoutLayer(0.3, 'Name', 'Dropout_2') % Add dropout
    
    fullyConnectedLayer(2, 'Name', 'Output_Scores')
    softmaxLayer('Name', 'Probabilities')
    classificationLayer('Name', 'Loss')
];

% Let's visualize the more powerful architecture we just designed
fprintf('Displaying the new adaptive network architecture...\n');
analyzeNetwork(layers);


%% PHASE 2: GENERATE THE AGGREGATED "MEGA" DATASET
fprintf('\n--- PHASE 2: Generating and Aggregating Data for All Tau Values ---\n');

% Pre-allocate cell arrays to hold data from each loop iteration
X_cell = cell(1, length(tau_train_set));
Y_cell = cell(1, length(tau_train_set));

% Use parfor for potential speedup on multi-core machines if you have the Parallel Computing Toolbox
for i = 1:length(tau_train_set)
    current_tau = tau_train_set(i);
    fprintf('  -> Generating %d samples for tau = %.1f...\n', N_per_tau, current_tau);
    [X_cell{i}, Y_cell{i}] = generate_adaptive_data(N_per_tau, current_tau, SNR_for_training, window_len, num_feedback_taps);
end

% Combine the data from all cell arrays into two large matrices
X_mega_dataset = cat(1, X_cell{:});
Y_mega_dataset = cat(1, Y_cell{:});

fprintf('\nTotal training samples generated: %d\n', size(X_mega_dataset, 1));


%% PHASE 3: TRAINING THE SINGLE ADAPTIVE NETWORK
fprintf('\n--- PHASE 3: Training the Generalist Network ---\n');

% --- Partition the aggregated data ---
Y_categorical = categorical(Y_mega_dataset);
cv = cvpartition(size(X_mega_dataset, 1), 'HoldOut', 0.1); % Use 10% for validation
X_train = X_mega_dataset(cv.training, :);
Y_train = Y_categorical(cv.training, :);
X_val = X_mega_dataset(cv.test, :);
Y_val = Y_categorical(cv.test, :);

% --- Define Training Options ---
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...           % Give the larger network enough time to learn
    'MiniBatchSize', 256, ...        % A larger batch size is common for larger datasets
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 500, ...  % Check validation performance periodically
    'GradientThreshold', 1, ...      % Helps prevent exploding gradients
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch');     % IMPORTANT: Shuffle data to mix taus in each batch

% --- Train the Network ---
adaptive_model = trainNetwork(X_train, Y_train, layers, options);

%% PHASE 4: SAVE THE FINAL GENERALIST MODEL
fprintf('\n--- PHASE 4: Saving the final, adaptive detector model ---\n');
save('final_adaptive_detector.mat', 'adaptive_model');
fprintf('Model saved successfully to "final_adaptive_detector.mat".\n');
fprintf('You may now run the corresponding testing script.\n');


%% --- HELPER FUNCTION ---
function [X, Y] = generate_adaptive_data(N, tau, SNR, win_len, num_taps)
    % This helper function generates a complete dataset for ONE specific tau
    fprintf('    (Generating signal for tau=%.1f...)\n', tau);
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    half_win=floor(win_len/2);
    
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_up = upsample(symbols, round(sps*tau));
    
    % Generate the received signal after the matched filter
    rxMF = conv(conv(tx_up, h), h);
    rxMF = awgn(rxMF, SNR, 'measured'); % Add noise
    
    % Find the precise delay for alignment
    delay = finddelay(tx_up, rxMF);
    
    % Pre-allocate for speed
    X = zeros(N, win_len + num_taps + 1);
    Y = zeros(N, 1);
    
    % Loop to create each enhanced data vector
    for i = (num_taps + 1):N
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            win = rxMF(loc-half_win:loc+half_win).';
            % Use teacher-forcing with TRUE past symbols
            hist = symbols(i-1:-1:i-num_taps).';
            
            % Create the final adaptive input vector
            X(i, :) = [win, hist, tau];
            Y(i) = bits(i);
        end
    end
    
    % Clean up any unused rows
    valid_rows = any(X,2);
    X = X(valid_rows,:);
    Y = Y(valid_rows,:);
    fprintf('    (Generated %d valid samples for tau=%.1f)\n', size(X,1), tau);
end