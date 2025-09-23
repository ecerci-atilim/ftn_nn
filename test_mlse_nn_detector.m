% =========================================================================
% FINAL TESTING SCRIPT: The "Delayed Decision" Investigator
% =========================================================================
% GOAL:
% To test the "Oracle" network using a realistic "delayed decision"
% strategy, where a noisy preview of the future is used as an input.
% Includes a UI to select the model file.
% =========================================================================

clear, clc,% close all

%% PHASE 1: LOAD THE TRAINED MODEL USING A UI
fprintf('--- PHASE 1: Loading the MLSE-like NN model ---\n');

% Open a UI dialog to select the model file from the 'mat' folder
[file, path] = uigetfile('mat/*.mat', 'Select a Trained Model File');
if isequal(file, 0)
    error('User canceled model selection. Exiting.');
end
model_filename = fullfile(path, file);

try
    load(model_filename); % This will load 'mlse_model'
    fprintf('Model "%s" loaded successfully.\n', file);
catch ME
    error('Could not load the selected model file.');
end

% Extract tau from the filename for the test
tok = regexp(file, 'tau_(\d+)', 'tokens');
tau = str2double(tok{1}{1}) / 10;
fprintf('Testing with tau = %.1f, based on filename.\n', tau);


%% PHASE 2: SETUP THE BER SIMULATION
fprintf('\n--- PHASE 2: Setting up the BER simulation ---\n');

SNR_range = 0:10;
ber_results = zeros(size(SNR_range));

%% PHASE 3: THE MAIN BER TESTING LOOP
fprintf('\n--- PHASE 3: Starting the final BER simulation loop ---\n');

% parfor snridx = 1:length(SNR_range)
for snridx = 1:length(SNR_range)
    current_SNR = SNR_range(snridx);
    fprintf('  -> Testing at SNR = %d dB...\n', current_SNR);
    [errors, bits_sim] = run_mlse_ber_test(mlse_model, tau, current_SNR);
    ber_results(snridx) = errors / bits_sim;
    fprintf('    --> (tau=%.1f) SNR: %d dB, BER: %.4e (%d errors in %d bits)\n', tau, current_SNR, ber_results(snridx), errors, bits_sim);
end

%% PHASE 4: PLOT THE FINAL RESULTS
fprintf('\n--- PHASE 4: Plotting the performance curve ---\n');
figure;
semilogy(SNR_range, ber_results, 'b-d', 'LineWidth', 2, 'MarkerSize', 8);
grid on; grid minor;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend(sprintf('MLSE-like NN, tau = %.1f', tau));
title('Final Performance of the "Oracle" NN Detector');
ylim([1e-6 0.5]);


%% --- VERIFIED HELPER FUNCTION for MLSE-NN TESTING ---
function [num_errors, num_bits] = run_mlse_ber_test(model, tau, SNR)
    % This function implements the robust "delayed decision" testing strategy.
    
    % Get parameters from the model itself
    input_len = model.Layers(1).InputSize;
    num_past_taps = 4;
    num_future_taps = 2;
    win_len = input_len - num_past_taps - num_future_taps;
    half_win = floor(win_len/2);

    num_errors = 0;
    num_bits = 0;
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while num_errors < 100 && num_bits < 4e5
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        symbols = 1-2*bits;
        
        tx_up = upsample(symbols, round(sps*tau));
        rxSignal = awgn(conv(tx_up, h), SNR, 'measured');
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        decision_history = zeros(num_past_taps, 1);
        
        for i = (num_past_taps + 1):(N_batch - num_future_taps)
            center_loc = round((i-1)*sps*tau) + 1 + delay;
            
            if (center_loc > half_win) && (center_loc + half_win <= length(rxMF))
                current_window = rxMF(center_loc - half_win : center_loc + half_win).';
                past_symbols = (1-2*decision_history).';
                
                % --- ROBUST FUTURE SYMBOL APPROXIMATION ---
                future_samples = zeros(1, num_future_taps);
                for j = 1:num_future_taps
                    future_loc = round((i+j-1)*sps*tau) + 1 + delay;
                    if future_loc <= length(rxMF)
                        future_samples(j) = rxMF(future_loc);
                    end
                end
                
                % Use a simple slicer (sign) as a noisy estimate of future symbols
                future_symbols_approx = sign(future_samples);
                future_symbols_approx(future_symbols_approx == 0) = 1;

                % Assemble the final vector for the test
                input_vector = [current_window, past_symbols, future_symbols_approx];
                
                prediction = classify(model, input_vector);
                prediction_bit = str2double(string(prediction));
                
                if prediction_bit ~= bits(i)
                    num_errors = num_errors + 1;
                end
                
                decision_history = [prediction_bit; decision_history(1:end-1)];
                num_bits = num_bits + 1;
            end
        end
    end
end