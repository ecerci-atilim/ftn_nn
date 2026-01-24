clear; clc; close all;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% Dosya seçimi
[files, path] = uigetfile('mat/comparison/*_results.mat', 'MultiSelect', 'on');
if isequal(files, 0), error('No file'); end
if ischar(files), files = {files}; end

% İlk dosyadan model isimlerini al
load(fullfile(path, files{1}));
[sel, ok] = listdlg('ListString', names, 'SelectionMode', 'multiple', ...
    'PromptString', 'Çizilecek modelleri seç:', 'ListSize', [200 150]);
if ~ok, return; end

fprintf('Seçilen modeller: %s\n', strjoin(names(sel), ', '));
fprintf('Dosya sayısı: %d\n\n', length(files));

% Renk paleti (τ değerleri için)
tau_colors = {'b', 'r', 'g', 'm', 'c', 'k', [0.5 0.5 0], [1 0.5 0]};
markers = {'o', 's', 'd', '^', 'v', 'p', 'x', '+'};

figure('Position', [100 100 1000 600]);

legend_entries = {};
h_plots = [];

% AWGN theory (bir kez çiz)
load(fullfile(path, files{1}));
h_awgn = semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2);
legend_entries{end+1} = 'AWGN Theory';
h_plots(end+1) = h_awgn;
hold on;

for f = 1:length(files)
    load(fullfile(path, files{f}));
    
    col = tau_colors{mod(f-1, length(tau_colors)) + 1};
    
    for idx = 1:length(sel)
        m = sel(idx);
        mrk = markers{mod(idx-1, length(markers)) + 1};
        
        if ~any(isnan(ber(m,:)))
            h = semilogy(SNR_range, ber(m,:), ...
                'Color', col, ...
                'Marker', mrk, ...
                'LineStyle', '-', ...
                'LineWidth', 1.5, ...
                'MarkerSize', 7);
            h_plots(end+1) = h;
            legend_entries{end+1} = sprintf('%s ($\\tau=%.1f$)', names{m}, tau);
        end
    end
end

grid on
grid minor
xlabel('$E_b/N_0 (dB)$', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
legend(h_plots, legend_entries, 'Location', 'southwest', 'FontSize', 8);
title('FTN Detection Comparison', 'FontSize', 13);
ylim([1e-6 1]);
% xlim([SNR_range(1) SNR_range(end)]);

% Kaydet
if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/combined_comparison.png');
saveas(gcf, 'figures/combined_comparison.fig');

fprintf('Figür kaydedildi.\n');