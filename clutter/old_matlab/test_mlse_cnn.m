% =========================================================================
% FINAL TESTING SCRIPT: The High-Performance Decision Feedback CNN (DF-CNN)
% =========================================================================

clear, clc, close all

%% --- PHASE 1: LOAD THE TRAINED DF-CNN MODEL ---
[file, path] = uigetfile('mat/df_cnn*.mat', 'Select a Trained DF-CNN Model');
if isequal(file, 0), error('User canceled model selection.'); end
model_filename = fullfile(path, file);
load(model_filename); % loads 'df_cnn_model'
tok = regexp(file, 'tau_(\d+)', 'tokens');
tau = str2double(tok{1}{1}) / 10;
fprintf('Testing with model "%s" for tau = %.1f\n', file, tau);

%% --- PHASE 2: SETUP THE BER SIMULATION ---
SNR_range_dB = 0:1:10;
ber_results = zeros(size(SNR_range_dB));

%% --- PHASE 3: THE MAIN BER TESTING LOOP ---
for snridx = 1:length(SNR_range_dB)
    current_SNR_dB = SNR_range_dB(snridx);
    [errors, bits_sim] = run_df_cnn_ber_test(df_cnn_model, tau, current_SNR_dB);
    ber_results(snridx) = errors / bits_sim;
    fprintf('  (tau=%.1f) SNR: %2d dB  ->  BER: %.4e (%d errors in %d bits)\n', tau, current_SNR_dB, ber_results(snridx), errors, bits_sim);
end

%% --- PHASE 4: PLOT FINAL RESULTS AGAINST THEORETICAL LIMIT ---
SNR_lin = 10.^(SNR_range_dB/10);
ber_theoretical = qfunc(sqrt(2*SNR_lin));

figure;
semilogy(SNR_range_dB, ber_theoretical, 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical BPSK (No ISI)');
hold on;
semilogy(SNR_range_dB, ber_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('DF-CNN, tau = %.1f', tau));
hold off;
grid on; grid minor;
xlabel('SNR (Eb/N0) [dB]'); ylabel('Bit Error Rate (BER)');
legend('show', 'Location', 'southwest');
title('High-Performance DF-CNN vs. Theoretical Limit');
ylim([1e-7 0.5]);

%% --- HELPER FUNCTION for DF-CNN TESTING ---
function [e,b] = run_df_cnn_ber_test(model, tau, SNR_dB)
    input_len = 35; % Hard-code to the known size
    num_taps = 4;
    win_len = input_len - num_taps;
    half_win = floor(win_len/2);
    e=0; b=0;
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while e < 100 && b < 1e6
        N_batch = 10000;
        bits = randi([0 1], N_batch, 1);
        symbols = 1-2*bits;
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        
        pwr = mean(txSignal.^2); txSignal = txSignal / sqrt(pwr);
        snr_lin = 10^(SNR_dB/10); noise_var = 1 / (2*snr_lin);
        noise = sqrt(noise_var) * randn(size(txSignal));
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