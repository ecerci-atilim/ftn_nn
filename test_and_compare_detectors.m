% =========================================================================
% CAPSTONE ASSIGNMENT: Performance Showdown (FINAL, ALIGNMENT-CORRECTED)
% =========================================================================
% GOAL:
% To test our detectors using a robust, automated alignment method
% (`finddelay`) to ensure the data seen during testing is perfectly
% aligned, matching the implicit alignment learned during training.
% =========================================================================

clear, clc, close all

set(groot,'defaultAxesTickLabelInterpreter','latex');      % Interpreter definition for axes ticks of figures
set(groot,'defaulttextinterpreter','latex');               % Interpreter definition for default strings casted on figures
set(groot,'defaultLegendInterpreter','latex');             % Interpreter definitions for default legend strings displayed on figures

%% PHASE 1: LOAD MODELS AND SETUP
fprintf('PHASE 1: Loading models and setting up parameters...\n');
try
    load('cnn_detector.mat');
    load('final_df_fnn_detector.mat');
catch ME
    error('Could not load model files. Please ensure they are in the MATLAB path.');
end

% --- Simulation Parameters ---
SNR_range = 0:10;
sps = 10; tau = 0.8; beta = 0.3; span = 6;
h = rcosdesign(beta, span, sps, 'sqrt');
cnn_win_len = 31; % Hard-code the known correct value
df_win_len = 35;
num_feedback_taps = 4;
half_win = floor(cnn_win_len/2);

% --- Pre-allocate result arrays ---
ber_cnn = zeros(size(SNR_range));
ber_df_fnn = zeros(size(SNR_range));

%% PHASE 2: THE MAIN BER TESTING LOOP
fprintf('\nPHASE 2: Starting robust BER simulation loop...\n');

for snridx = 1:length(SNR_range)
    current_SNR = SNR_range(snridx);
    fprintf('\n--- Testing at SNR = %d dB ---\n', current_SNR);

    % --- Reset counters ---
    errors_cnn = 0; bits_cnn = 0;
    errors_df_fnn = 0; bits_df_fnn = 0;
    
    while (errors_cnn < 100 && bits_cnn < 2e5) % Safety break
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        symbols = 1 - 2*bits;
        
        % --- Generate Signals ---
        tx_upsampled = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_upsampled, h);
        rxSignal = awgn(txSignal, current_SNR, 'measured');
        rxMF = conv(rxSignal, h);

        % --- ROBUST ALIGNMENT USING FINDDELAY ---
        % We find the delay between the upsampled sequence and the matched
        % filter output. This automatically accounts for all filter delays.
        delay = finddelay(tx_upsampled, rxMF);
        
        % --- Test CNN ---
        xdata_cnn = zeros(N_batch, cnn_win_len);
        for i = 1:N_batch
            % The center of the symbol is at the upsampled location + delay
            center_loc = round((i-1)*sps*tau) + 1 + delay;
            if (center_loc > half_win) && (center_loc + half_win <= length(rxMF))
                xdata_cnn(i, :) = rxMF(center_loc - half_win : center_loc + half_win);
            end
        end
        valid_rows = any(xdata_cnn, 2);
        X_test_cnn = num2cell(xdata_cnn(valid_rows,:), 2);
        Y_test_bits = bits(valid_rows);
        Y_pred_cnn = classify(ftn_detector_cnn, X_test_cnn);
        errors_cnn = errors_cnn + sum(Y_pred_cnn ~= categorical(Y_test_bits));
        bits_cnn = bits_cnn + length(Y_test_bits);

        % --- Test DF-NN ---
        decision_history = zeros(num_feedback_taps, 1);
        for i = (num_feedback_taps + 1):N_batch
            center_loc = round((i-1)*sps*tau) + 1 + delay;
            if (center_loc + half_win <= length(rxMF))
                current_window = rxMF(center_loc - half_win : center_loc + half_win).';
                feedback_symbols = 1 - 2*decision_history;
                input_vector = [current_window, feedback_symbols.'];
                prediction = classify(ftn_detector_df_fnn, input_vector);
                if prediction ~= categorical(bits(i))
                    errors_df_fnn = errors_df_fnn + 1;
                end
                decision_history = [str2double(string(prediction)); decision_history(1:end-1)];
            end
        end
        bits_df_fnn = bits_df_fnn + (N_batch - num_feedback_taps);
        
        fprintf('  Batch update -> CNN Errors: %d/%d | DF-NN Errors: %d/%d\n', errors_cnn, bits_cnn, errors_df_fnn, bits_df_fnn);
    end
    
    ber_cnn(snridx) = errors_cnn / bits_cnn;
    ber_df_fnn(snridx) = errors_df_fnn / bits_df_fnn;
    fprintf('--> Final BER -> CNN: %.4e | DF-NN: %.4e\n', ber_cnn(snridx), ber_df_fnn(snridx));
end

%% PHASE 3: PLOT THE FINAL RESULTS
fprintf('\nPHASE 3: Plotting the final performance comparison...\n');
figure;
semilogy(SNR_range, ber_cnn, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'CNN Detector');
hold on;
semilogy(SNR_range, ber_df_fnn, 'gd-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'DF-NN');
grid on; grid minor;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('Location', 'southwest');
% title('Performance Showdown (Correctly Aligned)');