% =========================================================================
% UNIVERSAL DF-CNN TESTING: Multi-Modulation BER Testing
% =========================================================================
% Tests your universal DF-CNN against theoretical limits for any modulation
% =========================================================================

clear, clc, close all

%% LOAD MODEL
[file, path] = uigetfile('mat/universal_df/*.mat', 'Select a Trained Universal DF-CNN Model');
if isequal(file, 0), error('User canceled model selection.'); end
model_filename = fullfile(path, file);
load(model_filename); % loads model and parameters

fprintf('Testing Universal DF-CNN: %d-%s, phase=%.3f, tau=%.1f\n', ...
    modulation_order, upper(modulation_type), phase_offset, tau);

%% SETUP BER SIMULATION
SNR_range_dB = 0:1:12;
ser_results = zeros(size(SNR_range_dB));
ber_results = zeros(size(SNR_range_dB));

%% MAIN TESTING LOOP
for snridx = 1:length(SNR_range_dB)
    current_SNR_dB = SNR_range_dB(snridx);
    [symbol_errors, bit_errors, total_symbols, total_bits] = run_universal_df_test(universal_df_model, modulation_type, modulation_order, phase_offset, tau, current_SNR_dB);
    
    ser_results(snridx) = symbol_errors / total_symbols;
    ber_results(snridx) = bit_errors / total_bits;
    
    fprintf('SNR: %2d dB -> SER: %.4e, BER: %.4e (%d/%d symbols, %d/%d bits)\n', ...
        current_SNR_dB, ser_results(snridx), ber_results(snridx), ...
        symbol_errors, total_symbols, bit_errors, total_bits);
end

%% PLOT RESULTS AGAINST THEORETICAL LIMITS
figure('Position', [100, 100, 900, 600]);

% Calculate theoretical BER
if strcmpi(modulation_type, 'psk')
    if modulation_order == 2
        ber_theoretical = qfunc(sqrt(2 * 10.^(SNR_range_dB/10)));
    else
        ber_theoretical = berawgn(SNR_range_dB, 'psk', modulation_order, 'nondiff');
    end
elseif strcmpi(modulation_type, 'qam')
    ber_theoretical = berawgn(SNR_range_dB, 'qam', modulation_order);
end

% Plot BER curves
semilogy(SNR_range_dB, ber_theoretical, 'k--', 'LineWidth', 2.5, 'DisplayName', sprintf('Theoretical %d-%s', modulation_order, upper(modulation_type)));
hold on;
semilogy(SNR_range_dB, ber_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', sprintf('Universal DF-CNN (τ=%.1f)', tau));
hold off;

grid on; grid minor;
xlabel('SNR (E_b/N_0) [dB]', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);
legend('show', 'Location', 'southwest', 'FontSize', 11);
title(sprintf('Universal DF-CNN: %d-%s, τ=%.1f, phase=%.2fπ', modulation_order, upper(modulation_type), tau, phase_offset/pi), 'FontSize', 13);
ylim([1e-7 0.5]);

% Save results
results_file = strrep(file, '.mat', '_ber_results.mat');
save(fullfile([path,'\results'], results_file), 'SNR_range_dB', 'ber_results', 'ser_results', 'ber_theoretical');
fprintf('Results saved to: %s\n', results_file);

%% HELPER FUNCTION - CORRECTED TESTING WITH PROPER Eb/N0
function [sym_errors, bit_errors, total_symbols, total_bits] = run_universal_df_test(model, mod_type, M, phase, tau, SNR_dB)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Test parameters
    input_len = 66; % window_len*2 + num_feedback_taps  
    num_taps = 4;
    win_len = 31;
    half_win = floor(win_len/2);
    
    % Determine if modulation is real
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    sym_errors = 0; bit_errors = 0; total_symbols = 0; total_bits = 0;
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while sym_errors < 100 && total_symbols < 1e6
        N_batch = 10000;
        
        % Generate symbols
        symbol_indices = randi([0 M-1], N_batch, 1);
        symbols = constellation(symbol_indices + 1);
        
        % Channel simulation  
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        pwr = mean(abs(txSignal).^2); txSignal = txSignal / sqrt(pwr);
        
        % CORRECTED NOISE GENERATION: Convert Eb/N0 to Es/N0
        snr_eb_n0 = 10^(SNR_dB/10);        % Eb/N0 in linear scale
        snr_es_n0 = k * snr_eb_n0;         % Convert to Es/N0
        signal_power = mean(abs(txSignal).^2);
        noise_power = signal_power / snr_es_n0;
        
        % Generate appropriate noise type
        if is_real_modulation  % BPSK with real symbols
            noise = sqrt(noise_power) * randn(size(txSignal));
        else  % Complex modulations (QPSK, QAM)
            noise = sqrt(noise_power/2) * (randn(size(txSignal)) + 1j*randn(size(txSignal)));
        end
        
        rxSignal = txSignal + noise;
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        % Decision feedback history
        decision_history = zeros(num_taps, 1);
        
        for i = (num_taps + 1):N_batch
            loc = round((i-1)*sps*tau) + 1 + delay;
            if (loc > half_win) && (loc + half_win <= length(rxMF))
                % Extract signal features
                win_complex = rxMF(loc-half_win:loc+half_win);
                
                if is_real_modulation
                    % For BPSK: use real samples twice to maintain dimension compatibility
                    win_features = [real(win_complex(:))', real(win_complex(:))'];
                else
                    % For complex modulations: proper I/Q features
                    win_features = [real(win_complex(:))', imag(win_complex(:))'];
                end
                
                % Combine with decision feedback
                test_input = [win_features, decision_history'];
                
                % NN prediction
                pred_probs = predict(model, test_input);
                [~, pred_idx] = max(pred_probs);
                predicted_symbol_idx = pred_idx - 1; % Convert to 0-based
                
                % Symbol error
                if predicted_symbol_idx ~= symbol_indices(i)
                    sym_errors = sym_errors + 1;
                end
                
                % Bit errors (for BER calculation)
                true_bits = de2bi(symbol_indices(i), k, 'left-msb');
                pred_bits = de2bi(predicted_symbol_idx, k, 'left-msb');
                bit_errors = bit_errors + sum(true_bits ~= pred_bits);
                
                % Update decision history
                decision_history = [predicted_symbol_idx; decision_history(1:end-1)];
                
                total_symbols = total_symbols + 1;
                total_bits = total_bits + k;
            end
        end
    end
end

function constellation = generate_constellation(M, type, phase)
    if strcmpi(type, 'psk')
        p = 0:M-1;
        constellation = exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam')
        k = log2(M);
        if mod(k,2) ~= 0
            error('QAM order must be a power of 4');
        end
        n = sqrt(M);
        vals = -(n-1):2:(n-1);
        [X,Y] = meshgrid(vals, vals);
        constellation = (X + 1j*Y) * exp(1j*phase);
        constellation = constellation(:);
    end
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end