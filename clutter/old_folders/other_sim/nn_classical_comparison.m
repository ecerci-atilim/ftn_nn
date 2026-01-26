clear; clc; close all;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Load classical results
load('classical_results.mat');

%% Load NN results
[files, path] = uigetfile('../mat/comparison/*_results.mat', 'MultiSelect', 'on');
if isequal(files, 0), error('No file'); end
if ischar(files), files = {files}; end

%% Her tau için ayrı grafik
for f = 1:length(files)
    load(fullfile(path, files{f}));
    
    % Tau index for classical results
    t_idx = find(abs(tau_values - tau) < 0.01);
    if isempty(t_idx), continue; end
    
    figure('Position', [100 + f*50, 100, 800, 550]);
    
    ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));
    
    % AWGN Theory
    semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
    hold on;
    
    % NN models
    semilogy(SNR_range, ber(6,:), 'b-s', 'LineWidth', 2, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'b', 'DisplayName', 'Window+DF (NN)');
    
    semilogy(SNR_range, ber(4,:), 'm-d', 'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', 'Neighbors+DF (NN)');
    
    % Classical detectors
    semilogy(SNR_range, ber_bcjr(t_idx,:), 'r-o', 'LineWidth', 2, 'MarkerSize', 7, ...
        'DisplayName', 'M-BCJR');
    
    semilogy(SNR_range, ber_kse(t_idx,:), 'g-^', 'LineWidth', 2, 'MarkerSize', 7, ...
        'DisplayName', 'SSSgbKSE');
    
    grid on;
    xlabel('$E_b/N_0$ (dB)', 'FontSize', 12);
    ylabel('BER', 'FontSize', 12);
    legend('Location', 'southwest', 'FontSize', 10);
    % title(sprintf('NN vs Classical Detectors (τ = %.1f)', tau), 'FontSize', 13);
    ylim([1e-6 1]);
    xlim([SNR_range(1) SNR_range(end)]);
    
    % % Save
    % saveas(gcf, sprintf('figures/comparison_tau%02d.png', tau*10));
    % saveas(gcf, sprintf('figures/comparison_tau%02d.fig', tau*10));
    
    % Print table
    idx10 = find(SNR_range == 10);
    if ~isempty(idx10)
        fprintf('\nτ = %.1f @ 10 dB\n', tau);
        fprintf('Window+DF:    %.2e\n', ber(6, idx10));
        fprintf('Neighbors+DF: %.2e\n', ber(4, idx10));
        fprintf('M-BCJR:       %.2e\n', ber_bcjr(t_idx, idx10));
        fprintf('SSSgbKSE:     %.2e\n', ber_kse(t_idx, idx10));
    end
end
