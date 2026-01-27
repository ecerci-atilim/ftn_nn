% =========================================================================
% FINAL TESTING SCRIPT: The Decision Feedback CNN (DF-CNN)
% =========================================================================
% GOAL:
% To test the high-performance DF-CNN and compare its BER curve directly
% against the theoretical limit for uncoded BPSK in AWGN.
% =========================================================================

clear, clc, close all

%% --- PHASE 1: LOAD THE TRAINED DF-CNN MODEL ---
fprintf('--- PHASE 1: Loading the trained DF-CNN model...\n');
[file, path] = uigetfile('mat/df_cnn*.mat', 'Select a Trained DF-CNN Model');
if isequal(file, 0), error('User canceled model selection.'); end
model_filename = fullfile(path, file);
try
    load(model_filename); % loads 'df_cnn_model'
    fprintf('Model "%s" loaded successfully.\n', file);
catch ME
    error('Could not load the selected model file.');
end
tok = regexp(file, 'tau_(\d+)', 'tokens');
tau = str2double(tok{1}{1}) / 10;
fprintf('Testing with tau = %.1f, based on filename.\n', tau);

%% --- PHASE 2: SETUP THE BER SIMULATION ---
SNR_range_dB = 0:1:10;
ber_results = zeros(size(SNR_range_dB));

%% --- PHASE 3: THE MAIN BER TESTING LOOP ---
fprintf('\n--- PHASE 3: Starting BER simulation loop...\n');
parfor snridx = 1:length(SNR_range_dB)
    current_SNR_dB = SNR_range_dB(snridx);
    [errors, bits_sim] = run_df_cnn_ber_test(df_cnn_model, tau, current_SNR_dB);
    ber_results(snridx) = errors / bits_sim;
    fprintf('  (tau=%.1f) SNR: %2d dB  ->  BER: %.4e (%d errors in %d bits)\n', tau, current_SNR_dB, ber_results(snridx), errors, bits_sim);
end

%% --- PHASE 4: PLOT FINAL RESULTS AGAINST THEORETICAL LIMIT ---
fprintf('\n--- PHASE 4: Plotting final performance curve...\n');
% --- Calculate Theoretical BPSK BER ---
SNR_lin = 10.^(SNR_range_dB/10);
ber_theoretical = qfunc(sqrt(2*SNR_lin));

figure;
semilogy(SNR_range_dB, ber_theoretical, 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical BPSK (No ISI)');
hold on;
semilogy(SNR_range_dB, ber_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('DF-CNN, tau = %.1f', tau));
hold off;
grid on; grid minor;
xlabel('SNR (Eb/N0) [dB]');
ylabel('Bit Error Rate (BER)');
legend('show', 'Location', 'southwest');
title('DF-CNN Performance vs. Theoretical Limit');
ylim([1e-6 0.5]);


%% --- VERIFIED HELPER FUNCTION for DF-CNN TESTING ---
function [e,b] = run_df_cnn_ber_test(model, tau, SNR_dB)
    input_len = model.Layers(1).InputSize;
    num_taps = 4;
    win_len = input_len - num_taps;
    half_win = floor(win_len/2);
    e=0; b=0; 
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while e < 100 && b < 5e5
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        symbols = 1-2*bits;
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        
        % --- MANUAL NOISE GENERATION (MUST MATCH TRAINING) ---
        pwr = mean(txSignal.^2);
        txSignal = txSignal / sqrt(pwr);
        snr_lin = 10^(SNR_dB/10);
        noise_variance = 1 / (2*snr_lin);
        noise = sqrt(noise_variance) * randn(size(txSignal));
        rxSignal = txSignal + noise;
        
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
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