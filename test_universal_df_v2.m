clear, clc, close all

[file, path] = uigetfile('mat/universal_df/*.mat', 'Select a Trained Universal DF-CNN Model');
if isequal(file, 0), error('User canceled model selection.'); end
model_filename = fullfile(path, file);
load(model_filename);

fprintf('Testing Universal DF-CNN: %d-%s, phase=%.3f, tau=%.1f\n', ...
    modulation_order, upper(modulation_type), phase_offset, tau);

SNR_range_dB = 0:1:12;
ser_results = zeros(size(SNR_range_dB));
ber_results = zeros(size(SNR_range_dB));

min_errors = 100;
max_symbols = 1e6;

fprintf('\nTest configuration: min_errors=%d, max_symbols=%d\n', min_errors, max_symbols);

tic;
for snridx = 1:length(SNR_range_dB)
    current_SNR_dB = SNR_range_dB(snridx);
    t_start = tic;
    [symbol_errors, bit_errors, total_symbols, total_bits] = run_universal_df_test_fast(universal_df_model, modulation_type, modulation_order, phase_offset, tau, current_SNR_dB, min_errors, max_symbols);
    t_elapsed = toc(t_start);
    
    ser_results(snridx) = symbol_errors / total_symbols;
    ber_results(snridx) = bit_errors / total_bits;
    
    fprintf('SNR: %2d dB -> SER: %.4e, BER: %.4e (%d/%d symbols, %d/%d bits) [%.1f sec]\n', ...
        current_SNR_dB, ser_results(snridx), ber_results(snridx), ...
        symbol_errors, total_symbols, bit_errors, total_bits, t_elapsed);
end
total_time = toc;
fprintf('\nTotal simulation time: %.1f seconds (%.1f min)\n', total_time, total_time/60);

figure('Position', [100, 100, 900, 600]);

if strcmpi(modulation_type, 'psk')
    if modulation_order == 2
        ber_theoretical = qfunc(sqrt(2 * 10.^(SNR_range_dB/10)));
    else
        ber_theoretical = berawgn(SNR_range_dB, 'psk', modulation_order, 'nondiff');
    end
elseif strcmpi(modulation_type, 'qam')
    ber_theoretical = berawgn(SNR_range_dB, 'qam', modulation_order);
end

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

if ~exist(fullfile(path, 'results'), 'dir')
    mkdir(fullfile(path, 'results'));
end
results_file = strrep(file, '.mat', '_ber_results.mat');
save(fullfile(path, 'results', results_file), 'SNR_range_dB', 'ber_results', 'ser_results', 'ber_theoretical');
fprintf('Results saved to: %s\n', fullfile(path, 'results', results_file));

function [sym_errors, bit_errors, total_symbols, total_bits] = run_universal_df_test_fast(model, mod_type, M, phase, tau, SNR_dB, min_errors, max_symbols)
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    num_taps = 4;
    win_len = 31;
    half_win = floor(win_len/2);
    batch_predict_size = 256;
    
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    sym_errors = 0; bit_errors = 0; total_symbols = 0; total_bits = 0;
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    snr_eb_n0 = 10^(SNR_dB/10);
    snr_es_n0 = k * snr_eb_n0;
    
    N_batch = 50000;
    
    while sym_errors < min_errors && total_symbols < max_symbols
        symbol_indices = randi([0 M-1], N_batch, 1);
        symbols = constellation(symbol_indices + 1);
        
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        pwr = mean(abs(txSignal).^2); 
        txSignal = txSignal / sqrt(pwr);
        
        signal_power = mean(abs(txSignal).^2);
        noise_power = signal_power / snr_es_n0;
        
        if is_real_modulation
            noise = sqrt(noise_power) * randn(size(txSignal));
        else
            noise = sqrt(noise_power/2) * (randn(size(txSignal)) + 1j*randn(size(txSignal)));
        end
        
        rxSignal = txSignal + noise;
        rxMF = conv(rxSignal, h);
        delay = finddelay(tx_up, rxMF);
        
        decision_history = zeros(num_taps, 1);
        
        input_buffer = [];
        idx_buffer = [];
        
        for i = (num_taps + 1):N_batch
            loc = round((i-1)*sps*tau) + 1 + delay;
            if (loc > half_win) && (loc + half_win <= length(rxMF))
                win_complex = rxMF(loc-half_win:loc+half_win);
                
                if is_real_modulation
                    win_features = [real(win_complex(:))', real(win_complex(:))'];
                else
                    win_features = [real(win_complex(:))', imag(win_complex(:))'];
                end
                
                test_input = [win_features, decision_history'];
                
                input_buffer = [input_buffer; test_input];
                idx_buffer = [idx_buffer; i];
                
                if size(input_buffer, 1) >= batch_predict_size || i == N_batch
                    pred_probs = predict(model, input_buffer);
                    [~, pred_idx] = max(pred_probs, [], 2);
                    predicted_symbol_idx = pred_idx - 1;
                    
                    for j = 1:length(idx_buffer)
                        curr_idx = idx_buffer(j);
                        curr_pred = predicted_symbol_idx(j);
                        
                        if curr_pred ~= symbol_indices(curr_idx)
                            sym_errors = sym_errors + 1;
                        end
                        
                        true_bits = de2bi(symbol_indices(curr_idx), k, 'left-msb');
                        pred_bits = de2bi(curr_pred, k, 'left-msb');
                        bit_errors = bit_errors + sum(true_bits ~= pred_bits);
                        
                        total_symbols = total_symbols + 1;
                        total_bits = total_bits + k;
                    end
                    
                    decision_history = [predicted_symbol_idx(end); decision_history(1:end-1)];
                    
                    input_buffer = [];
                    idx_buffer = [];
                    
                    if sym_errors >= min_errors
                        return;
                    end
                else
                    decision_history = [0; decision_history(1:end-1)];
                end
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