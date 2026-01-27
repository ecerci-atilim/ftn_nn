% =========================================================================
% ASSIGNMENT #10: Building and Training the Decision Feedback Neural Network
% =========================================================================
% SCIENTIFIC GOAL:
% To prove the hypothesis that providing a neural network with explicit
% historical information (past symbol decisions) will allow it to learn a
% more effective model for mitigating FTN Intersymbol Interference.
%
% METHODOLOGY:
% We will use "Teacher Forcing" during training, where the network is fed
% the 100% correct past symbols. This provides the ideal learning signal for
% the network to master the art of ISI cancellation.
% =========================================================================

clear, clc, close all

%% PHASE 1: THE EXPERIMENTAL DESIGN (ARCHITECTURE)
% Here, we use our domain knowledge to design a network specifically for
% our enhanced input data structure.

fprintf('PHASE 1: Designing the Network Architecture...\n');

% --- Network Hyperparameters ---
window_len = 31;         % The number of signal samples in our window
num_feedback_taps = 4;   % The number of past decisions we will feed back
enhanced_input_len = window_len + num_feedback_taps; % Total input size = 35

% --- The Architecture Definition ---
df_fnn_layers = [
    
    % Layer 1: The Input Port
    % Defines the shape for our 31 signal samples + 4 history taps.
    featureInputLayer(enhanced_input_len, 'Name', 'Input (Signal + History)')

    % Layer 2 & 3: First Hidden Layer & Activation
    % This layer finds the first level of patterns from the 35 inputs.
    % It has (35 * 128) + 128 = 4608 learnable parameters.
    fullyConnectedLayer(128, 'Name', 'Hidden_Layer_1')
    reluLayer('Name', 'Activation_1')

    % Layer 4 & 5: Second Hidden Layer & Activation
    % This layer refines the patterns from the first hidden layer.
    % It has (128 * 64) + 64 = 8256 learnable parameters.
    fullyConnectedLayer(64, 'Name', 'Hidden_Layer_2')
    reluLayer('Name', 'Activation_2')

    % Layer 6: The Final Voting Booth
    % Takes the 64 refined features and produces two scores: one for "Bit 0"
    % and one for "Bit 1".
    fullyConnectedLayer(2, 'Name', 'Output_Scores')
    
    % Layer 7 & 8: The Training Mechanism
    % Converts scores to probabilities and calculates the error (loss) that
    % drives the entire learning process.
    softmaxLayer('Name', 'Probabilities')
    classificationLayer('Name', 'Loss_and_Output')
];

% --- Visualize Your Creation ---
fprintf('Displaying the network architecture. Close the window to continue.\n');
analyzeNetwork(df_fnn_layers);


%% PHASE 2: THE EXPERIMENT (GENERATING "TEACHER-FORCED" DATA)
% This is where your communications engineering expertise is critical.
% We create a perfect, controlled simulation of the FTN channel to serve as
% the training ground for our network.

fprintf('\nPHASE 2: Generating Teacher-Forced Training Data...\n');

% --- Simulation Parameters ---
N = 50000;                  % Number of symbols to simulate for training
SNR_for_training = 15;      % Train at a high, clean SNR
sps = 10;                   % Samples per symbol
tau = 0.7;                  % FTN acceleration factor (80% of Nyquist period)
beta = 0.3;                 % Rolloff factor for the RRC filter
span = 6;                   % Filter span in symbols

% --- Generate the Base Signal ---
bits = randi([0 1], N, 1);
symbols = 1 - 2*bits; % BPSK modulation: 0 -> +1, 1 -> -1
h = rcosdesign(beta, span, sps, 'sqrt');

tx_upsampled = upsample(symbols, round(sps*tau));
txSignal = conv(tx_upsampled, h);
rxSignal = awgn(txSignal, SNR_for_training, 'measured');
rxMF = conv(rxSignal, h); % Apply matched filter

% --- Data Generation Loop ---
xdata_enhanced = zeros(N, enhanced_input_len);
ydata_enhanced = zeros(N, 1);
total_delay_win = floor(window_len/2);

% Calculate the correct start index for the first symbol's center
pulse_delay = (length(h)-1)/2;
start_idx = 2*pulse_delay + 1;

for i = (num_feedback_taps + 1):N
    % Get the sampling instant for the i-th symbol
    ploc = round(start_idx + (i-1)*sps*tau);

    % Ensure the window is within the bounds of the received signal
    if (ploc - total_delay_win > 0) && (ploc + total_delay_win <= length(rxMF))
        % 1. Extract the physical signal from your controlled experiment
        current_window = rxMF(ploc - total_delay_win : ploc + total_delay_win);

        % 2. Provide the "teacher's hint": the known ground-truth history
        past_symbols = symbols(i-1 : -1 : i-num_feedback_taps);

        % 3. Create the enhanced input vector
        xdata_enhanced(i, :) = [current_window.', past_symbols.']; % Ensure both are row vectors
        ydata_enhanced(i) = bits(i);
    end
end

% --- Clean up the dataset ---
% Remove the rows at the start and end that were not filled
valid_rows = (ydata_enhanced ~= 0) | all(xdata_enhanced ~= 0, 2);
xdata_enhanced = xdata_enhanced(valid_rows, :);
ydata_enhanced = ydata_enhanced(valid_rows, :);
fprintf('Generated %d high-quality training samples.\n', size(xdata_enhanced, 1));


%% PHASE 3: THE LEARNING PROCESS

fprintf('\nPHASE 3: Training the Decision Feedback Network...\n');

% --- Partition Data into Training and Validation Sets ---
Y_categorical = categorical(ydata_enhanced);
cv = cvpartition(size(xdata_enhanced, 1), 'HoldOut', 0.2);

X_train = xdata_enhanced(cv.training, :);
Y_train = Y_categorical(cv.training, :);
X_val = xdata_enhanced(cv.test, :);
Y_val = Y_categorical(cv.test, :);

% --- Define Training Options ---
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 50, ...
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% --- Train the Network ---
% This command initiates the learning process. The network will iterate
% through the data, compare its predictions to the true labels, calculate
% the error, and adjust its internal weights to minimize that error.
ftn_detector_df_fnn = trainNetwork(X_train, Y_train, df_fnn_layers, options);

%% PHASE 4: SAVING THE RESULT OF YOUR EXPERIMENT

fprintf('\nPHASE 4: Saving the final, trained detector model...\n');
fname = sprintf("df_fnn_detector_tau_%03d.mat", tau*100);
save(fname, 'ftn_detector_df_fnn');
fprintf('Model saved successfully to final_df_fnn_detector.mat.\n');
fprintf('\nNEXT STEP: Use this saved model in a separate BER testing script to evaluate its performance.\n');