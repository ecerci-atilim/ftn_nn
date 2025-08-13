% =========================================================================
% FINAL EXPERIMENT: Generating Performance Curves for an Ensemble of Specialists
% =========================================================================
% SCIENTIFIC GOAL:
% To achieve the highest possible performance for each tau value by training
% a dedicated, specialist neural network for each one. This script will
% train each specialist and then test it to generate a final family of
% high-performance BER curves, proving the validity of this approach.
% =========================================================================

clear, clc, close all

%% --- SETUP ---
tau_set = [0.5, 0.7, 0.9];
SNR_range = 0:2:10;
ber_results = zeros(length(tau_set), length(SNR_range));

%% --- MAIN LOOP: TRAIN AND TEST ONE SPECIALIST AT A TIME ---

for tau_idx = 1:length(tau_set)
    current_tau = tau_set(tau_idx);
    fprintf('\n======================================================\n');
    fprintf('--- Processing Specialist for tau = %.1f ---\n', current_tau);
    fprintf('======================================================\n');

    % --- 1. TRAIN THE SPECIALIST ---
    fprintf('\n(1) Training specialist model...\n');
    
    % Parameters
    N_train = 50000; SNR_train = 15; win_len = 31; num_taps = 4;
    input_len = win_len + num_taps;

    % Define architecture
    layers = [
    featureInputLayer(input_len)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
        
    % Generate training data (this uses the teacher-forcing helper)
    [x_train, y_train] = generate_training_data(N_train, current_tau, SNR_train, win_len, num_taps);
    Y_cat_train = categorical(y_train);
    
    % Use a small portion of the start of the data for validation during training
    val_count = min(5000, floor(length(y_train)*0.1));
    
    % Training options
    options = trainingOptions('adam', 'MaxEpochs', 12, 'MiniBatchSize', 128, ...
        'ValidationData', {x_train(1:val_count,:), Y_cat_train(1:val_count,:)}, ...
        'Verbose', false, 'Plots', 'none');
        
    % Train the specialist model
    specialist_model = trainNetwork(x_train, Y_cat_train, layers, options);
    fprintf('Training complete for tau = %.1f specialist.\n', current_tau);

    % --- 2. TEST THE SPECIALIST ---
    fprintf('\n(2) Testing specialist model across all SNRs...\n');
    ber_curve = zeros(1, length(SNR_range));

    parfor snr_idx = 1:length(SNR_range) % Use parfor for speed if available
        current_SNR = SNR_range(snr_idx);
        % Call the verified BER test helper function
        [errors, bits_sim] = run_ber_test(specialist_model, current_tau, current_SNR, win_len, num_taps);
        ber_curve(snr_idx) = errors / bits_sim;
        fprintf('  (tau=%.1f) SNR: %2d dB  ->  BER: %.4e (%d errors in %d bits)\n', current_tau, current_SNR, ber_curve(snr_idx), errors, bits_sim);
    end
    ber_results(tau_idx, :) = ber_curve;
end

%% --- 3. PLOT RESULTS ---
fprintf('\n(3) Plotting final results...\n');
figure;
hold on;
colors = ['r', 'g', 'b'];
markers = ['s', 'd', 'o'];
for i = 1:length(tau_set)
    semilogy(SNR_range, ber_results(i,:), [colors(i) markers(i) '-'], 'LineWidth', 2, ...
        'DisplayName', sprintf('Specialist Detector, tau = %.1f', tau_set(i)));
end
hold off;
grid on; grid minor;
xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)');
legend('show', 'Location', 'southwest');
title('Performance of Specialist NN Detectors');
ylim([1e-6 0.5]); % Adjust y-axis to see very low BERs

%% --- HELPER FUNCTIONS (VERIFIED) ---
function [x,y] = generate_training_data(N, tau, SNR, win_len, num_taps)
    % Generates teacher-forced data for training
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_upsampled = upsample(symbols, round(sps*tau));
    rxMF = conv(conv(tx_upsampled, h), h);
    rxMF = awgn(rxMF, SNR, 'measured');
    delay = finddelay(tx_upsampled, rxMF);
    x = zeros(N, win_len + num_taps);
    y = zeros(N, 1);
    half_win = floor(win_len/2);
    for i = (num_taps + 1):N
        center_loc = round((i-1)*sps*tau) + 1 + delay;
        if (center_loc > half_win) && (center_loc + half_win <= length(rxMF))
            x(i, :) = [rxMF(center_loc-half_win:center_loc+half_win).', symbols(i-1:-1:i-num_taps).'];
            y(i) = bits(i);
        end
    end
    valid_rows = any(x,2);
    x = x(valid_rows,:);
    y = y(valid_rows);
end

function [e,b] = run_ber_test(model, tau, SNR, win_len, num_taps)
    % Runs a sequential BER test for one SNR point
    e=0; b=0; half_win=floor(win_len/2);
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    while e < 100 && b < 4e5 % Increased bit limit to reach lower BERs
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        tx_upsampled = upsample(1-2*bits, round(sps*tau));
        rxSignal = awgn(conv(tx_upsampled, h), SNR, 'measured');
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_upsampled, rxMF);
        hist = zeros(num_taps, 1);
        for i = (num_taps + 1):N_batch
            loc = round((i-1)*sps*tau)+1+delay;
            if (loc > half_win) && (loc+half_win <= length(rxMF))
                win = rxMF(loc-half_win:loc+half_win).';
                fb = (1-2*hist).';
                pred = classify(model, [win, fb]);
                if pred ~= categorical(bits(i)); e=e+1; end
                hist = [str2double(string(pred)); hist(1:end-1)];
                b=b+1;
            end
        end
    end
end