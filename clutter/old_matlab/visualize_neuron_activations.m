% =========================================================================
% XAI SCRIPT: Visualizing the "Brain" of the DF-NN
% =========================================================================
% GOAL:
% To probe our trained Decision Feedback Neural Network by feeding it
% specific, handcrafted inputs and visualizing the activation levels of its
% internal neurons. This helps us understand what each layer has learned.
% =========================================================================

clear, clc, close all

%% PHASE 1: LOAD THE TRAINED MODEL AND PARAMETERS
fprintf('--- PHASE 1: Loading the trained DF-NN model ---\n');
try
    load('final_df_fnn_detector.mat');
catch ME
    error('Could not load final_df_fnn_detector.mat.');
end

% --- System Parameters (must match training) ---
sps = 10; tau = 0.8; beta = 0.3; span = 6;
h = rcosdesign(beta, span, sps, 'sqrt');
cnn_win_len = 31;
num_feedback_taps = 4;
half_win = floor(cnn_win_len/2);

%% PHASE 2: CRAFTING ARTIFICIAL, PERFECT INPUTS
fprintf('--- PHASE 2: Crafting perfect input signals for testing ---\n');

% We will create a small sequence and extract the center pulse
N_craft = 15; % Must be > span
center_symbol_idx = round(N_craft/2);

% --- Test Case 1: A perfect "+1" bit (symbol 0) with no interference ---
symbols_test1 = zeros(1, N_craft);
symbols_test1(center_symbol_idx) = 1; % A single '1' symbol
bits_test1 = (1 - symbols_test1)/2;
feedback_test1 = ones(1, num_feedback_taps) * -1; % History of all '+1's is simple

% --- Test Case 2: A perfect "-1" bit (symbol 1) with no interference ---
symbols_test2 = zeros(1, N_craft);
symbols_test2(center_symbol_idx) = -1; % A single '-1' symbol
bits_test2 = (1 - symbols_test2)/2;
feedback_test2 = ones(1, num_feedback_taps); % History of all '-1's

% --- Test Case 3: A "+1" bit with strong ISI from a preceding "-1" ---
symbols_test3 = zeros(1, N_craft);
symbols_test3(center_symbol_idx) = 1;
symbols_test3(center_symbol_idx - 1) = -1; % The interfering symbol
bits_test3 = (1 - symbols_test3)/2;
feedback_test3 = [1, -1, -1, -1]; % History: k-1 was -1, k-2 was +1, etc.

% --- Helper function to generate the signal window ---
function window = generate_window(symbols, h, sps, tau, center_idx, win_len)
    tx_upsampled = upsample(symbols, round(sps*tau));
    txMF = conv(conv(tx_upsampled, h), h); % Transmit and matched filter
    delay = finddelay(tx_upsampled, txMF);
    center_loc = round((center_idx-1)*sps*tau) + 1 + delay;
    half_w = floor(win_len/2);
    window = txMF(center_loc - half_w : center_loc + half_w);
end

% Generate the noiseless signal windows for our test cases
window1 = generate_window(symbols_test1, h, sps, tau, center_symbol_idx, cnn_win_len);
window2 = generate_window(symbols_test2, h, sps, tau, center_symbol_idx, cnn_win_len);
window3 = generate_window(symbols_test3, h, sps, tau, center_symbol_idx, cnn_win_len);

% Create the final, enhanced input vectors
input_vector1 = [window1, feedback_test1];
input_vector2 = [window2, feedback_test2];
input_vector3 = [window3, feedback_test3];

%% PHASE 3: EXTRACTING ACTIVATIONS
fprintf('--- PHASE 3: Propagating inputs and extracting activations ---\n');

% The `activations` function is MATLAB's tool for this "neuroscience".
% We specify the input data and the names of the layers we want to observe.
activations1_L1 = activations(ftn_detector_df_fnn, input_vector1, 'Activation_1');
activations1_L2 = activations(ftn_detector_df_fnn, input_vector1, 'Activation_2');

activations2_L1 = activations(ftn_detector_df_fnn, input_vector2, 'Activation_1');
activations2_L2 = activations(ftn_detector_df_fnn, input_vector2, 'Activation_2');

activations3_L1 = activations(ftn_detector_df_fnn, input_vector3, 'Activation_1');
activations3_L2 = activations(ftn_detector_df_fnn, input_vector3, 'Activation_2');

fprintf('Activations extracted successfully.\n');

%% PHASE 4: VISUALIZATION
fprintf('--- PHASE 4: Generating plots ---\n');

figure('Position', [100, 100, 1200, 800]);

% --- Plot the Input Signals ---
subplot(3,3,1);
plot(window1, 'b-o'); title('Input Signal: Perfect "+1"'); grid on; axis tight;
subplot(3,3,4);
plot(window2, 'r-o'); title('Input Signal: Perfect "-1"'); grid on; axis tight;
subplot(3,3,7);
plot(window3, 'g-o'); title('Input Signal: "+1" with ISI'); grid on; axis tight;
xlabel('Sample Index');

% --- Plot Activations for Hidden Layer 1 (128 neurons) ---
subplot(3,3,2);
imagesc(activations1_L1); colorbar;
title('Layer 1 Activations for "+1"'); ylabel('Neuron Index');
subplot(3,3,5);
imagesc(activations2_L1); colorbar;
title('Layer 1 Activations for "-1"'); ylabel('Neuron Index');
subplot(3,3,8);
imagesc(activations3_L1); colorbar;
title('Layer 1 Activations for "+1" with ISI'); ylabel('Neuron Index');
xlabel('Neuron Index');

% --- Plot Activations for Hidden Layer 2 (64 neurons) ---
subplot(3,3,3);
imagesc(activations1_L2); colorbar;
title('Layer 2 Activations for "+1"');
subplot(3,3,6);
imagesc(activations2_L2); colorbar;
title('Layer 2 Activations for "-1"');
subplot(3,3,9);
imagesc(activations3_L2); colorbar;
title('Layer 2 Activations for "+1" with ISI');
xlabel('Neuron Index');

sgtitle('Neural Network Activation Visualization', 'FontSize', 16, 'FontWeight', 'bold');

% --- Command Window Analysis ---
% Let's find some neurons that behave differently
diff_L1 = abs(activations1_L1 - activations3_L1);
[~, most_changed_neuron_L1] = max(diff_L1);
fprintf('\n--- Analysis ---\n');
fprintf('The neuron in Layer 1 that changed MOST between the perfect "+1" and the ISI case is Neuron #%d\n', most_changed_neuron_L1);

diff_L2 = abs(activations1_L2 - activations3_L2);
[~, most_changed_neuron_L2] = max(diff_L2);
fprintf('The neuron in Layer 2 that changed MOST between the perfect "+1" and the ISI case is Neuron #%d\n', most_changed_neuron_L2);