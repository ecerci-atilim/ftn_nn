% =========================================================================
% COMPARE MULTIPLE MODELS: Plot BER curves on same figure
% =========================================================================
% Loads saved result files and plots them together for comparison
% Uses LaTeX formatting for professional publication-quality plots
% =========================================================================

clear, clc, close all

%% CONFIGURATION
results_folder = 'C:\Users\GPX PRO\Desktop\ftn_nn\mat\universal_df\results';

% Define colors and markers for different models
colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k'];
markers = ['o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '*'];

%% SELECT RESULT FILES TO COMPARE
[files, path] = uigetfile(fullfile(results_folder, '*_ber_results.mat'), ...
    'Select Result Files to Compare (Ctrl/Shift for multiple)', 'MultiSelect', 'on');

if isequal(files, 0)
    error('No files selected. Exiting.');
end

% Handle single file selection
if ischar(files)
    files = {files};
end

fprintf('=== COMPARING %d MODELS ===\n', length(files));

%% LOAD ALL RESULTS
results = cell(length(files), 1);
for i = 1:length(files)
    load(fullfile(path, files{i}));
    results{i}.SNR = SNR_range_dB;
    results{i}.BER = ber_results;
    results{i}.SER = ser_results;
    results{i}.mod_type = modulation_type;
    results{i}.mod_order = modulation_order;
    results{i}.tau = tau;
    results{i}.phase = phase_offset;
    results{i}.filename = files{i};
    
    fprintf('%d. %s: %d-%s, tau=%.1f\n', i, files{i}, modulation_order, upper(modulation_type), tau);
end

%% CREATE COMPARISON PLOT WITH LATEX FORMATTING
fig = figure('Position', [100, 100, 1200, 700]);

% Plot theoretical curve (use first model's parameters as reference)
ref_mod_type = results{1}.mod_type;
ref_mod_order = results{1}.mod_order;
SNR_plot = results{1}.SNR;

if strcmpi(ref_mod_type, 'psk')
    if ref_mod_order == 2
        ber_theoretical = qfunc(sqrt(2 * 10.^(SNR_plot/10)));
    else
        ber_theoretical = berawgn(SNR_plot, 'psk', ref_mod_order, 'nondiff');
    end
elseif strcmpi(ref_mod_type, 'qam')
    ber_theoretical = berawgn(SNR_plot, 'qam', ref_mod_order);
end

semilogy(SNR_plot, ber_theoretical, 'k--', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Theoretical %d-%s', ref_mod_order, upper(ref_mod_type)));
hold on;

% Plot each model's results
for i = 1:length(results)
    color_idx = mod(i-1, length(colors)) + 1;
    marker_idx = mod(i-1, length(markers)) + 1;
    
    % Create legend label with LaTeX formatting
    label = sprintf('$\\tau=%.1f$, $\\phi=%.2f\\pi$', results{i}.tau, results{i}.phase/pi);
    
    semilogy(results{i}.SNR, results{i}.BER, ...
        'Color', colors(color_idx), ...
        'Marker', markers(marker_idx), ...
        'LineWidth', 2, ...
        'MarkerSize', 7, ...
        'DisplayName', label);
end

hold off;
grid on; grid minor;

% Set LaTeX interpreter for all text
xlabel('SNR ($E_b/N_0$) [dB]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Bit Error Rate (BER)', 'Interpreter', 'latex', 'FontSize', 14);

% Legend with LaTeX interpreter
lgd = legend('show', 'Location', 'southwest', 'FontSize', 11);
set(lgd, 'Interpreter', 'latex');

% Title with LaTeX formatting
title_str = sprintf('Model Comparison: %d-%s Detection', ref_mod_order, upper(ref_mod_type));
title(title_str, 'Interpreter', 'latex', 'FontSize', 15);

ylim([1e-7 0.5]);
xlim([min(SNR_plot) max(SNR_plot)]);

% Set axes font size
set(gca, 'FontSize', 12);

%% ADD PERFORMANCE TABLE WITH LATEX FORMATTING
% Create text annotation showing key performance metrics
table_text = sprintf('\\textbf{Performance at 10dB:}\n');
for i = 1:length(results)
    idx_10dB = find(results{i}.SNR == 10, 1);
    if ~isempty(idx_10dB)
        table_text = sprintf('%s$\\tau=%.1f$: BER=$%.2e$\n', table_text, results{i}.tau, results{i}.BER(idx_10dB));
    end
end

annotation('textbox', [0.15, 0.15, 0.25, 0.2], 'String', table_text, ...
    'Interpreter', 'latex', 'FontSize', 11, 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'FitBoxToText', 'on');

%% SAVE COMPARISON PLOT
save_filename = sprintf('comparison_%d_models_%s', length(files), datestr(now, 'yyyymmdd_HHMMSS'));
saveas(fig, fullfile(results_folder, [save_filename, '.png']));
saveas(fig, fullfile(results_folder, [save_filename, '.fig']));
fprintf('\nComparison plot saved to: %s\n', save_filename);

%% GENERATE COMPARISON TABLE
fprintf('\n=== DETAILED COMPARISON TABLE ===\n');
fprintf('%-30s | Tau  | Phase    | BER@6dB    | BER@10dB   | BER@12dB\n', 'Model');
fprintf('%-30s-|------|----------|------------|------------|------------\n', repmat('-',1,30));

for i = 1:length(results)
    % Extract filename without path and extension
    [~, name, ~] = fileparts(results{i}.filename);
    if length(name) > 30
        name = [name(1:27), '...'];
    end
    
    % Get BER at specific SNRs
    ber_6 = results{i}.BER(results{i}.SNR == 6);
    ber_10 = results{i}.BER(results{i}.SNR == 10);
    ber_12 = results{i}.BER(results{i}.SNR == 12);
    
    if isempty(ber_6), ber_6 = NaN; end
    if isempty(ber_10), ber_10 = NaN; end
    if isempty(ber_12), ber_12 = NaN; end
    
    fprintf('%-30s | %.1f | %.2fπ | %.3e | %.3e | %.3e\n', ...
        name, results{i}.tau, results{i}.phase/pi, ber_6, ber_10, ber_12);
end

fprintf('\n=== COMPARISON COMPLETE ===\n');