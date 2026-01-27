% --- SCRIPT 2: TEST_BER_CURVE.m ---
clear, clc

% --- Load the one, pre-trained model ---
fprintf('Loading pre-trained CNN detector...\n');
load('cnn_detector.mat');

% --- Simulation Parameters ---
N = 1e4;
SNR_range = 0:10;
ber_results = zeros(size(SNR_range));
sps = 10;
tau = .8;
beta = .3;
span = 6;
window_len = 2*floor(3*sps/2)+1;
h = rcosdesign(beta, span, sps, 'sqrt');
pulse_loc = (length(h)-1)/2;
total_delay_win = floor(window_len/2);
xdata = zeros(N, window_len);
ydata = zeros(N, 1);

% --- Main BER Testing Loop ---
for snridx = 1:length(SNR_range)
    current_SNR = SNR_range(snridx);
    num_errors = 0;
    num_bits_simulated = 0;
    
    fprintf('Testing at SNR = %d dB...\n', current_SNR);
    
    % Loop until we count enough errors for statistical significance
    while num_errors < 100 && num_bits_simulated < 2e6
        bits = randi([0 1], N, 1);
        symbols = 1-2*bits;
        rxMF = conv(awgn(conv(upsample(symbols, tau*sps), h), current_SNR, 'measured'), h);
        % Generate a NEW BATCH of random data at the current SNR
        % (e.g., N_batch = 10000 bits)
        % [Your data generation code for N_batch symbols at current_SNR]
        start_idx = 2*pulse_loc + 1;
        ploc = start_idx;
        
        for i = 1 : N
            if (ploc - total_delay_win > 0) && (ploc + total_delay_win <= length(rxMF))
                xdata(i, :) = rxMF(ploc - total_delay_win : ploc + total_delay_win);
            end
            ploc = ploc + sps*tau;
        end
        ydata = bits;
        
        % Prepare the batch for classification
        X_test_batch = num2cell(xdata, 2);
        
        % Classify using the single, loaded network
        Y_pred = classify(ftn_detector_cnn, X_test_batch);
        
        % Accumulate errors and bits
        num_errors = num_errors + sum(Y_pred ~= categorical(ydata));
        num_bits_simulated = num_bits_simulated + length(ydata);
    end
    
    % Calculate the final BER for this SNR point
    ber_results(snridx) = num_errors / num_bits_simulated;
    fprintf('--> SNR: %d dB, Errors: %d, Bits: %d, BER: %.4e\n', current_SNR, num_errors, num_bits_simulated, ber_results(snridx));
end

% --- Final Plot ---
% [Your plotting code for the BER curve]