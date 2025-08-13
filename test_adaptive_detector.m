% =========================================================================
% SCRIPT 2: Test the Final, Verified Adaptive Detector
% =========================================================================
% GOAL:
% To generate the final BER performance curves for our adaptive model,
% using the now-verified testing logic.
% =========================================================================

clear, clc, close all

%% --- PHASE 1: LOAD THE ADAPTIVE DETECTOR ---
fprintf('--- PHASE 1: Loading the adaptive detector model...\n');
try
    load('final_adaptive_detector.mat');
    fprintf('Model "final_adaptive_detector.mat" loaded successfully.\n');
catch ME
    error('Could not load model file. Please run the training script first.');
end

%% --- PHASE 2: SETUP THE SIMULATION ---
fprintf('--- PHASE 2: Setting up multi-tau BER simulation...\n');

tau_test_set = [0.5, 0.7, 0.9];
SNR_range = 0:2:10;
ber_results = zeros(length(tau_test_set), length(SNR_range));

%% --- PHASE 3: MAIN TESTING LOOP ---
fprintf('--- PHASE 3: Starting final BER simulation...\n');

for tau_idx = 1:length(tau_test_set)
    current_tau = tau_test_set(tau_idx);
    fprintf('\n------------------------------------------\n');
    fprintf('--- Testing for tau = %.1f ---\n', current_tau);
    fprintf('------------------------------------------\n');
    ber_curve = zeros(1, length(SNR_range));

    for snr_idx = 1:length(SNR_range)
        current_SNR = SNR_range(snr_idx);
        % Call the verified BER test function
        [errors, bits_sim] = run_adaptive_ber_test(adaptive_model, current_tau, current_SNR);
        ber_curve(snr_idx) = errors / bits_sim;
        fprintf('  SNR: %2d dB  ->  BER: %.4e (%d errors in %d bits)\n', current_SNR, ber_curve(snr_idx), errors, bits_sim);
    end
    ber_results(tau_idx, :) = ber_curve;
end

%% --- PHASE 4: PLOT FINAL RESULTS ---
fprintf('\n--- PHASE 4: Plotting final performance curves...\n');
figure;
hold on;
colors = ['r', 'g', 'b']; markers = ['s', 'd', 'o'];
for i = 1:length(tau_test_set)
    semilogy(SNR_range, ber_results(i,:), [colors(i) markers(i) '-'], 'LineWidth', 2, ...
        'DisplayName', sprintf('Adaptive Model, tau = %.1f', tau_test_set(i)));
end
hold off;
grid on; grid minor;
xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)');
legend('show', 'Location', 'southwest');
title('Final Performance of the Tau-Adaptive Detector');
ylim([1e-5 0.5]);


%% --- VERIFIED HELPER FUNCTION ---
function [e,b] = run_adaptive_ber_test(model, tau, SNR)
    % This function uses the verified logic from the debug script.
    e=0; b=0; 
    win_len=31; num_taps=4; half_win=floor(win_len/2);
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while e < 100 && b < 3e5 % Increased safety break
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        tx_up = upsample(1-2*bits, round(sps*tau));
        rxSignal = awgn(conv(tx_up, h), SNR, 'measured');
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        hist = zeros(num_taps, 1);
        for i = (num_taps + 1):N_batch
            loc = round((i-1)*sps*tau)+1+delay;
            if (loc > half_win) && (loc+half_win <= length(rxMF))
                win = rxMF(loc-half_win:loc+half_win).';
                fb = (1-2*hist).';
                pred = classify(model, [win, fb, tau]);
                if pred ~= categorical(bits(i)); e=e+1; end
                hist = [str2double(string(pred)); hist(1:end-1)];
                b=b+1;
            end
        end
    end
end