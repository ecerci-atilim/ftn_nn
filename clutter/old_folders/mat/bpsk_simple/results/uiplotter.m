% =========================================================================
% FINAL ANALYSIS SCRIPT: BER Curve Comparison Tool
% =========================================================================

clear, clc, close all

gbk_7 = [0.1913
    0.1754
    0.1588
    0.1351
    0.1181
    0.0895
    0.0775
    0.0573
    0.0441
    0.0356
    0.0253
    0.0175
    0.0110
    0.0061
    0.0044
    0.0023
    0.0010]';
snr_gbk7 = 0:length(gbk_7)-1;

gbk_8 = [0.1744
    0.1569
    0.1295
    0.1006
    0.0767
    0.0609
    0.0393
    0.0250
    0.0156
    0.0095
    0.0051
    0.0021
    0.0009
    0.0003]';
snr_gbk8 = 0:length(gbk_8)-1;

gbk_9 = [0.1646
    0.1351
    0.1084
    0.0884
    0.0646
    0.0435
    0.0278
    0.0152
    0.0073
    0.0031
    0.0011]';
snr_gbk9 = 0:length(gbk_9)-1;

mbcjr_07 = [0.245250000000000
   0.215950000000000
   0.194150000000000
   0.174050000000000
   0.143850000000000
   0.121100000000000
   0.095771144278607
   0.065837320574163
   0.043822222222222
   0.027299270072993
   0.012285012285012
   0.006002490660025
   0.001812327506900
   0.000502174772637
   0.000116722456702
   0.000012700000000];
snr_mbcjr_07 = 0:length(mbcjr_07)-1;

mbcjr_08 = [0.171400000000000
   0.143100000000000
   0.118500000000000
   0.087950000000000
   0.067550000000000
   0.046069651741294
   0.029764150943396
   0.016626016260163
   0.006715328467153
   0.002857142857143
   0.001019230769231
   0.000232108317215
   0.000041282622468
   0.000005000000000
   0.000000400000000
                   0];
snr_mbcjr_08 = 0:length(mbcjr_08)-1;

mbcjr_09 = [0.167850000000000
   0.139200000000000
   0.107900000000000
   0.081300000000000
   0.062128712871287
   0.039567307692308
   0.025227272727273
   0.012852233676976
   0.006507936507937
   0.002090909090909
   0.000654290931223
   0.000181818181818
   0.000027600000000
   0.000005600000000
                   0
                   0];
snr_mbcjr_09 = 0:length(mbcjr_09)-1;

%% --- PHASE 1: USER FILE SELECTION ---
fprintf('--- PHASE 1: Please select the result files to plot ---\n');

[files, path] = uigetfile('*.mat', 'Select BER Result Files (.mat)', 'MultiSelect', 'on');

if isequal(files, 0)
    fprintf('User canceled file selection. Exiting.\n');
    return;
end

if ~iscell(files)
    files = {files};
end

fprintf('Loaded %d result file(s).\n', length(files));

%% --- PHASE 2: PREPARE THE PLOT ---
fprintf('--- PHASE 2: Preparing the comparison plot ---\n');

figure;
grid on; grid minor;

try
    first_file_data = load(fullfile(path, files{1}));
    if isfield(first_file_data, 'SNR_range')
        SNR_range_dB = first_file_data.SNR_range;
        SNR_lin = 10.^(SNR_range_dB/10);
        ber_theoretical = qfunc(sqrt(2*SNR_lin));
        semilogy(SNR_range_dB, ber_theoretical, 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical BPSK');
        hold on
        fprintf('Theoretical BPSK curve plotted successfully.\n');
    else
        warning('The first file did not contain an "SNR_range" variable. Skipping theoretical curve.');
    end
catch ME
    % warning('Could not load data from the first file to plot theoretical curve. Error: %s', ME.message);
end

%% --- PHASE 3: LOOP, LOAD, AND PLOT EACH RESULT FILE ---
fprintf('--- PHASE 3: Parsing and plotting each result file ---\n');

colors = {'b', 'r', 'g', 'm', 'c', '#D95319', '#7E2F8E'};
markers = {'o', 'd', 's', '^', 'v', '*', 'x'};

for i = 1:length(files)
    current_file = files{i};
    fprintf('  -> Processing: %s\n', current_file);
    
    try
        file_data = load(fullfile(path, current_file));
        
        if ~isfield(file_data, 'ber_nn') || ~isfield(file_data, 'SNR_range')
            warning('Skipping file "%s" because it is missing "ber_nn" or "SNR_range".', current_file);
            continue;
        end
        
        % legend_str = 'Unknown';
        % 
        % tok = regexp(current_file, 'bpsk_df_tau(\d+)(_extfeat)?_results.mat', 'tokens');
        % 
        % if ~isempty(tok)
        %     tau_val_str = tok{1}{1};
        %     tau = str2double(tau_val_str) / 100;
        % 
        %     has_ext_feat = ~isempty(tok{1}{2});
        % 
        %     legend_str = sprintf('Tau = %.2f', tau);
        %     % if has_ext_feat
        %     %     legend_str = sprintf('Tau = %.2f (with Extra Features)', tau);
        %     % else
        %     %     legend_str = sprintf('Tau = %.2f (Signal Only)', tau);
        %     % end
        % else
        %     legend_str = strrep(current_file, '_', ' '); % Fallback legend
        % end
        
        style_idx = mod(i-1, length(colors)) + 1;
        plot_style = [colors{style_idx} '*-'];
        
        semilogy(file_data.SNR_range, file_data.ber_nn, plot_style, ...
            'LineWidth', 1.5);
            
    catch ME
        warning('Could not process file "%s". Error: %s', current_file, ME.message);
    end
end

semilogy(snr_mbcjr_07-3, mbcjr_07, 'bs:')
semilogy(snr_mbcjr_08-3, mbcjr_08, 'rs:')
semilogy(snr_mbcjr_09-3, mbcjr_09, 'gs:')
semilogy(snr_gbk7-3, gbk_7, 'bd--')
semilogy(snr_gbk8-3, gbk_8, 'rd--')
semilogy(snr_gbk9-3, gbk_9, 'gd--')

%% --- PHASE 4: FINALIZE THE PLOT ---
hold off;
xlabel('SNR (Eb/N0) [dB]');
ylabel('Bit Error Rate (BER)');
title('BER Performance Comparison');

legend('Theoretical BPSK',...
    'Multi-Sample ($\tau=0.7$)', 'Multi-Sample ($\tau=0.8$)', 'Multi-Sample ($\tau=09$)',...
    '32-BCJR ($\tau=0.7$)', '32-BCJR ($\tau=0.8$)', '32-BCJR ($\tau=0.9$)',...
    'SSSgb1SE ($\tau=0.7$)', 'SSSgb1SE ($\tau=0.8$)', 'SSSgb1SE ($\tau=0.9$)',...
    'Location', 'sw')

ylim([1e-6 0.5]);
xlim([min(SNR_range_dB), max(SNR_range_dB)]);
grid on
grid minor

fprintf('\n--- Plotting complete. ---\n');