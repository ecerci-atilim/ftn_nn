% =========================================================================
% BATCH UNIVERSAL DF-CNN TESTING: Test Multiple Models Overnight
% =========================================================================
% Allows selection of multiple models and tests them sequentially
% Saves results automatically to specified folder
% =========================================================================

clear, clc, close all

%% CONFIGURATION
results_folder = 'C:\Users\GPX PRO\Desktop\ftn_nn\mat\universal_df\results';
SNR_range_dB = 0:1:12;
min_errors = 20;        % Reduced from 100 for faster testing
max_symbols = 5e5;      % Maximum symbols to test per SNR point

% Create results folder if it doesn't exist
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
    fprintf('Created results folder: %s\n', results_folder);
end

%% SELECT MULTIPLE MODELS
[files, path] = uigetfile('mat/universal_df/*.mat', 'Select Models to Test (use Ctrl/Shift for multiple)', 'MultiSelect', 'on');
if isequal(files, 0)
    error('No models selected. Exiting.');
end

% Handle single file selection (uigetfile returns char instead of cell)
if ischar(files)
    files = {files};
end

fprintf('\n=== BATCH TESTING: %d models selected ===\n', length(files));
for i = 1:length(files)
    fprintf('  %d. %s\n', i, files{i});
end

%% MAIN BATCH TESTING LOOP
total_start_time = tic;

for model_idx = 1:length(files)
    fprintf('\n\n========================================\n');
    fprintf('TESTING MODEL %d/%d: %s\n', model_idx, length(files), files{model_idx});
    fprintf('========================================\n');
    
    model_start_time = tic;
    
    try
        % Load model
        model_filename = fullfile(path, files{model_idx});
        load(model_filename);
        
        fprintf('Configuration: %d-%s, phase=%.3f, tau=%.1f\n', ...
            modulation_order, upper(modulation_type), phase_offset, tau);
        
        % Initialize results
        ser_results = zeros(size(SNR_range_dB));
        ber_results = zeros(size(SNR_range_dB));
        test_times = zeros(size(SNR_range_dB));
        error_counts = zeros(size(SNR_range_dB));
        symbol_counts = zeros(size(SNR_range_dB));
        
        % Test at each SNR
        for snridx = 1:length(SNR_range_dB)
            current_SNR_dB = SNR_range_dB(snridx);
            snr_start = tic;
            
            [symbol_errors, bit_errors, total_symbols, total_bits] = ...
                run_universal_df_test_fast(universal_df_model, modulation_type, ...
                modulation_order, phase_offset, tau, current_SNR_dB, min_errors, max_symbols);
            
            test_times(snridx) = toc(snr_start);
            error_counts(snridx) = symbol_errors;
            symbol_counts(snridx) = total_symbols;
            
            ser_results(snridx) = symbol_errors / total_symbols;
            ber_results(snridx) = bit_errors / total_bits;
            
            fprintf('  SNR %2d dB: BER=%.4e (%d errors, %d symbols, %.1fs)\n', ...
                current_SNR_dB, ber_results(snridx), symbol_errors, total_symbols, test_times(snridx));
        end
        
        model_time = toc(model_start_time);
        fprintf('Model testing completed in %.1f minutes\n', model_time/60);
        
        % Generate and save plot
        fig = plot_ber_results(SNR_range_dB, ber_results, ser_results, ...
            modulation_type, modulation_order, tau, phase_offset);
        
        % Save results
        result_filename = strrep(files{model_idx}, '.mat', '_ber_results.mat');
        save(fullfile(results_folder, result_filename), 'SNR_range_dB', 'ber_results', ...
            'ser_results', 'test_times', 'error_counts', 'symbol_counts', ...
            'modulation_type', 'modulation_order', 'tau', 'phase_offset', 'model_time');
        
        % Save figure
        fig_filename = strrep(files{model_idx}, '.mat', '_ber_curve.png');
        saveas(fig, fullfile(results_folder, fig_filename));
        close(fig);
        
        fprintf('Results saved to: %s\n', result_filename);
        
    catch ME
        fprintf('ERROR testing model %s: %s\n', files{model_idx}, ME.message);
        fprintf('Continuing with next model...\n');
        continue;
    end
end

total_time = toc(total_start_time);
fprintf('\n\n========================================\n');
fprintf('BATCH TESTING COMPLETE\n');
fprintf('Total time: %.1f minutes (%.1f hours)\n', total_time/60, total_time/3600);
fprintf('Results saved to: %s\n', results_folder);
fprintf('========================================\n');

%% GENERATE SUMMARY REPORT
generate_summary_report(results_folder, files);

%% HELPER FUNCTIONS

function [sym_errors, bit_errors, total_symbols, total_bits] = run_universal_df_test_fast(model, mod_type, M, phase, tau, SNR_dB, min_errors, max_symbols)
    % Faster testing with reduced error threshold
    k = log2(M);
    constellation = generate_constellation(M, mod_type, phase);
    
    % Test parameters
    input_len = 66;
    num_taps = 4;
    win_len = 31;
    half_win = floor(win_len/2);
    
    is_real_modulation = (M == 2) && strcmpi(mod_type, 'psk');
    
    sym_errors = 0; bit_errors = 0; total_symbols = 0; total_bits = 0;
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    
    while sym_errors < min_errors && total_symbols < max_symbols
        N_batch = 5000;  % Smaller batches for better progress tracking
        
        % Generate symbols
        symbol_indices = randi([0 M-1], N_batch, 1);
        symbols = constellation(symbol_indices + 1);
        
        % Channel simulation  
        tx_up = upsample(symbols, round(sps*tau));
        txSignal = conv(tx_up, h);
        pwr = mean(abs(txSignal).^2); txSignal = txSignal / sqrt(pwr);
        
        % Corrected noise generation
        snr_eb_n0 = 10^(SNR_dB/10);
        snr_es_n0 = k * snr_eb_n0;
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
        
        % Decision feedback history
        decision_history = zeros(num_taps, 1);
        
        for i = (num_taps + 1):N_batch
            loc = round((i-1)*sps*tau) + 1 + delay;
            if (loc > half_win) && (loc + half_win <= length(rxMF))
                % Extract signal features
                win_complex = rxMF(loc-half_win:loc+half_win);
                
                if is_real_modulation
                    win_features = [real(win_complex(:))', real(win_complex(:))'];
                else
                    win_features = [real(win_complex(:))', imag(win_complex(:))'];
                end
                
                test_input = [win_features, decision_history'];
                
                % NN prediction
                pred_probs = predict(model, test_input);
                [~, pred_idx] = max(pred_probs);
                predicted_symbol_idx = pred_idx - 1;
                
                % Error counting
                if predicted_symbol_idx ~= symbol_indices(i)
                    sym_errors = sym_errors + 1;
                end
                
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

function fig = plot_ber_results(SNR_range_dB, ber_results, ser_results, mod_type, M, tau, phase)
    % Generate BER plot
    fig = figure('Position', [100, 100, 900, 600], 'Visible', 'off');
    
    % Calculate theoretical BER
    if strcmpi(mod_type, 'psk')
        if M == 2
            ber_theoretical = qfunc(sqrt(2 * 10.^(SNR_range_dB/10)));
        else
            ber_theoretical = berawgn(SNR_range_dB, 'psk', M, 'nondiff');
        end
    elseif strcmpi(mod_type, 'qam')
        ber_theoretical = berawgn(SNR_range_dB, 'qam', M);
    end
    
    % Plot
    semilogy(SNR_range_dB, ber_theoretical, 'k--', 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Theoretical %d-%s', M, upper(mod_type)));
    hold on;
    semilogy(SNR_range_dB, ber_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 7, ...
        'DisplayName', sprintf('Universal DF-CNN (τ=%.1f)', tau));
    hold off;
    
    grid on; grid minor;
    xlabel('SNR (E_b/N_0) [dB]', 'FontSize', 12);
    ylabel('Bit Error Rate (BER)', 'FontSize', 12);
    legend('show', 'Location', 'southwest', 'FontSize', 11);
    title(sprintf('Universal DF-CNN: %d-%s, τ=%.1f, phase=%.2fπ', M, upper(mod_type), tau, phase/pi), 'FontSize', 13);
    ylim([1e-7 0.5]);
end

function generate_summary_report(results_folder, model_files)
    % Generate a summary text file of all tests
    summary_file = fullfile(results_folder, 'batch_test_summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, '===============================================\n');
    fprintf(fid, 'BATCH TESTING SUMMARY\n');
    fprintf(fid, 'Date: %s\n', datestr(now));
    fprintf(fid, '===============================================\n\n');
    
    for i = 1:length(model_files)
        result_file = strrep(model_files{i}, '.mat', '_ber_results.mat');
        result_path = fullfile(results_folder, result_file);
        
        if exist(result_path, 'file')
            load(result_path);
            
            fprintf(fid, 'Model %d: %s\n', i, model_files{i});
            fprintf(fid, '  Configuration: %d-%s, tau=%.1f, phase=%.2fπ\n', ...
                modulation_order, upper(modulation_type), tau, phase_offset/pi);
            fprintf(fid, '  Testing time: %.1f minutes\n', model_time/60);
            fprintf(fid, '  BER at 10dB: %.4e\n', ber_results(SNR_range_dB==10));
            fprintf(fid, '\n');
        else
            fprintf(fid, 'Model %d: %s - FAILED\n\n', i, model_files{i});
        end
    end
    
    fclose(fid);
    fprintf('Summary report saved to: %s\n', summary_file);
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