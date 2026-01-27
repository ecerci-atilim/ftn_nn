set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

clear; clc; close all;

[files, path] = uigetfile('mat/comparison/*_results.mat', 'MultiSelect', 'on');
if isequal(files, 0), error('No file'); end
if ischar(files), files = {files}; end

fprintf('=== %d dosya seçildi ===\n\n', length(files));

% Ortak ayarlar
colors = {'c', 'c', 'm', 'm', 'b', 'g', 'k'};
markers = {'x', 'o', 'x', 'o', 's', 's', 'p'};
lines = {'-', '--', '-', '--', '-', '-', '-'};
% linewidths = [1.5, 1.5, 1.5, 2, 1.5, 2, 2.5];
linewidths = [1 1 1 1 1 1 1];

y_lim = [1e-6, 1];
% x_lim = [0, 14];

for f = 1:length(files)
    load(fullfile(path, files{f}));
    
    figure('Position', [100 + f*30, 100, 900, 550]);
    
    semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
    hold on;
    semilogy(SNR_range, ber_uncoded, 'r-^', 'LineWidth', 1.5, 'DisplayName', 'Uncoded FTN');
    
    for m = 1:7
        if ~any(isnan(ber(m,:)))
            semilogy(SNR_range, ber(m,:), [colors{m} lines{m} markers{m}], ...
                'LineWidth', linewidths(m), 'MarkerSize', 7, 'DisplayName', names{m});
        end
    end
    
    grid on;
    xlabel('$E_b/N_0$ (dB)', 'FontSize', 12);
    ylabel('BER', 'FontSize', 12);
    legend('Location', 'southwest', 'FontSize', 9);
    title(sprintf('$\\tau = %.1f$', tau), 'FontSize', 13);
    % xlim(x_lim);
    ylim(y_lim);
    
    % Export
    if ~exist('figures', 'dir'), mkdir('figures'); end
    saveas(gcf, sprintf('figures/ber_tau%02d.png', tau*10));
    saveas(gcf, sprintf('figures/ber_tau%02d.fig', tau*10));
    
    fprintf('τ=%.1f tamamlandı\n', tau);
end

fprintf('\n=== %d figür oluşturuldu ===\n', length(files));