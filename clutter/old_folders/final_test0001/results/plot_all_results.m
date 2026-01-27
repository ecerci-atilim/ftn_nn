%% Visualize All Results
% Combines reference and NN results for all tau values

clear; clc; close all;

%% Find available tau values
ref_files = dir('reference_tau*.mat');
res_files = dir('results_tau*.mat');

% Extract tau values
tau_ref = [];
for i = 1:length(ref_files)
    tok = regexp(ref_files(i).name, 'reference_tau(\d+\.\d+)\.mat', 'tokens');
    if ~isempty(tok)
        tau_ref(end+1) = str2double(tok{1}{1});
    end
end

tau_res = [];
for i = 1:length(res_files)
    tok = regexp(res_files(i).name, 'results_tau(\d+\.\d+)\.mat', 'tokens');
    if ~isempty(tok)
        tau_res(end+1) = str2double(tok{1}{1});
    end
end

tau_values = sort(unique([tau_ref, tau_res]));
fprintf('Found tau values: %s\n', num2str(tau_values));

%% Create figure directory
[~,~] = mkdir('figures');

%% Plot each tau
for tau = tau_values
    fprintf('\nPlotting tau = %.1f...\n', tau);
    
    fig = figure('Position', [100 100 900 600]);
    hold on;
    set(gca, 'YScale', 'log');
    legends = {};
    
    % Load reference
    ref_file = sprintf('reference_tau%.1f.mat', tau);
    if exist(ref_file, 'file')
        load(ref_file, 'results_ref');
        
        % Threshold
        if isfield(results_ref, 'threshold')
            semilogy(results_ref.threshold.SNR, results_ref.threshold.BER, ...
                     'k--', 'LineWidth', 1.5);
            legends{end+1} = 'Threshold';
        end
        
        % SSSgbKSE
        if isfield(results_ref, 'sss')
            semilogy(results_ref.sss.SNR, results_ref.sss.BER, ...
                     'k-v', 'LineWidth', 1.5, 'MarkerSize', 6);
            legends{end+1} = 'SSSgbKSE';
        end
        
        % M-BCJR
        if isfield(results_ref, 'bcjr')
            semilogy(results_ref.bcjr.SNR, results_ref.bcjr.BER, ...
                     'k-^', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'k');
            legends{end+1} = 'M-BCJR';
        end
    else
        fprintf('  No reference file for tau=%.1f\n', tau);
    end
    
    % Load NN results
    res_file = sprintf('results_tau%.1f.mat', tau);
    if exist(res_file, 'file')
        load(res_file, 'results', 'SNR_test');
        
        colors = lines(3);
        approaches = {'Neighbor', 'Fractional', 'Hybrid'};
        markers_nodf = {'o-', 's-', 'd-'};
        markers_df = {'o--', 's--', 'd--'};
        
        for app = 1:length(approaches)
            name_nodf = sprintf('%s_noDF', approaches{app});
            name_df = sprintf('%s_DF', approaches{app});
            
            if isfield(results, name_nodf)
                semilogy(SNR_test, results.(name_nodf).BER, markers_nodf{app}, ...
                         'Color', colors(app,:), 'LineWidth', 1.5, 'MarkerSize', 6);
                legends{end+1} = sprintf('%s (no DF)', approaches{app});
            end
            
            if isfield(results, name_df)
                semilogy(SNR_test, results.(name_df).BER, markers_df{app}, ...
                         'Color', colors(app,:), 'LineWidth', 1.5, 'MarkerSize', 6, ...
                         'MarkerFaceColor', colors(app,:));
                legends{end+1} = sprintf('%s (DF)', approaches{app});
            end
        end
    else
        fprintf('  No NN results file for tau=%.1f\n', tau);
    end
    
    grid on;
    xlabel('$E_b/N_0$ (dB)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('BER', 'Interpreter', 'latex', 'FontSize', 12);
    title(sprintf('BER Comparison: $\\tau = %.1f$', tau), ...
          'Interpreter', 'latex', 'FontSize', 14);
    legend(legends, 'Location', 'southwest', 'Interpreter', 'latex');
    ylim([1e-5 1]);
    xlim([0 14]);
    
    % Save
    saveas(fig, sprintf('figures/ber_tau%.1f.fig', tau));
    print(fig, sprintf('figures/ber_tau%.1f.png', tau), '-dpng', '-r150');
    print(fig, sprintf('figures/ber_tau%.1f.eps', tau), '-depsc2');
    fprintf('  Saved figures/ber_tau%.1f.{fig,png,eps}\n', tau);
end

%% Combined plot (all tau in subplots)
n_tau = length(tau_values);
if n_tau > 1
    fig_all = figure('Position', [50 50 400*min(n_tau,3) 350*ceil(n_tau/3)]);
    
    for t = 1:n_tau
        tau = tau_values(t);
        subplot(ceil(n_tau/3), min(n_tau,3), t);
        hold on;
        set(gca, 'YScale', 'log');
        
        % Reference
        ref_file = sprintf('reference_tau%.1f.mat', tau);
        if exist(ref_file, 'file')
            load(ref_file, 'results_ref');
            if isfield(results_ref, 'threshold')
                semilogy(results_ref.threshold.SNR, results_ref.threshold.BER, 'k--', 'LineWidth', 1);
            end
            if isfield(results_ref, 'sss')
                semilogy(results_ref.sss.SNR, results_ref.sss.BER, 'k-v', 'LineWidth', 1, 'MarkerSize', 4);
            end
            if isfield(results_ref, 'bcjr')
                semilogy(results_ref.bcjr.SNR, results_ref.bcjr.BER, 'k-^', 'LineWidth', 1.5, 'MarkerSize', 5);
            end
        end
        
        % NN
        res_file = sprintf('results_tau%.1f.mat', tau);
        if exist(res_file, 'file')
            load(res_file, 'results', 'SNR_test');
            colors = lines(3);
            approaches = {'Neighbor', 'Fractional', 'Hybrid'};
            
            for app = 1:length(approaches)
                name_df = sprintf('%s_DF', approaches{app});
                if isfield(results, name_df)
                    semilogy(SNR_test, results.(name_df).BER, 'o-', ...
                             'Color', colors(app,:), 'LineWidth', 1.5, 'MarkerSize', 4);
                end
            end
        end
        
        grid on;
        xlabel('$E_b/N_0$', 'Interpreter', 'latex');
        ylabel('BER', 'Interpreter', 'latex');
        title(sprintf('$\\tau = %.1f$', tau), 'Interpreter', 'latex');
        ylim([1e-5 1]);
        xlim([0 14]);
    end
    
    sgtitle('BER Comparison (DF only)', 'Interpreter', 'latex', 'FontSize', 14);
    
    saveas(fig_all, 'figures/ber_all_tau.fig');
    print(fig_all, 'figures/ber_all_tau.png', '-dpng', '-r150');
    fprintf('\nSaved figures/ber_all_tau.{fig,png}\n');
end

%% Summary table
fprintf('\n');
fprintf('==========================================\n');
fprintf('         BER Summary @ 10 dB\n');
fprintf('==========================================\n');
fprintf('tau    | Thresh  | SSS     | BCJR    | Nb+DF   | Frac+DF | Hyb+DF\n');
fprintf('-------|---------|---------|---------|---------|---------|--------\n');

for tau = tau_values
    row = sprintf('%.1f    |', tau);
    
    % Reference
    ref_file = sprintf('reference_tau%.1f.mat', tau);
    if exist(ref_file, 'file')
        load(ref_file, 'results_ref');
        idx = find(results_ref.threshold.SNR == 10, 1);
        if ~isempty(idx)
            row = [row, sprintf(' %.1e |', results_ref.threshold.BER(idx))];
            row = [row, sprintf(' %.1e |', results_ref.sss.BER(idx))];
            row = [row, sprintf(' %.1e |', results_ref.bcjr.BER(idx))];
        else
            row = [row, '   -     |   -     |   -     |'];
        end
    else
        row = [row, '   -     |   -     |   -     |'];
    end
    
    % NN
    res_file = sprintf('results_tau%.1f.mat', tau);
    if exist(res_file, 'file')
        load(res_file, 'results', 'SNR_test');
        idx = find(SNR_test == 10, 1);
        if ~isempty(idx)
            if isfield(results, 'Neighbor_DF')
                row = [row, sprintf(' %.1e |', results.Neighbor_DF.BER(idx))];
            else
                row = [row, '   -     |'];
            end
            if isfield(results, 'Fractional_DF')
                row = [row, sprintf(' %.1e |', results.Fractional_DF.BER(idx))];
            else
                row = [row, '   -     |'];
            end
            if isfield(results, 'Hybrid_DF')
                row = [row, sprintf(' %.1e', results.Hybrid_DF.BER(idx))];
            else
                row = [row, '   -    '];
            end
        else
            row = [row, '   -     |   -     |   -    '];
        end
    else
        row = [row, '   -     |   -     |   -    '];
    end
    
    fprintf('%s\n', row);
end

fprintf('==========================================\n');
fprintf('\nDone!\n');