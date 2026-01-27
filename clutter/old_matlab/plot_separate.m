%% FTN Results - Separate Figures
% Her grafik ayrı dosya olarak kaydedilir
%
% Emre Çerçi - Aralık 2025

clear; clc; close all;

%% Load Results
load('mat/comparison/full_results.mat');

fprintf('=== Generating Separate Figures ===\n\n');

%% AWGN Theory
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

%% Colors
c_awgn = [0 0 0];
c_th = [0.8 0 0];
c_sss = [0 0.6 0];
c_bcjr = [0 0.7 0.7];
c_nb = [0.8 0 0.8];
c_win = [0 0 0.8];

if ~exist('figures', 'dir'), mkdir('figures'); end

%% Figure 1: τ = 0.6
figure('Position', [100 100 700 550], 'Color', 'w');
t = 1;  % tau = 0.6

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th(t,:), '-^', 'Color', c_th, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_th, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss(t,:), '-v', 'Color', c_sss, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_sss, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_bcjr(t,:), '-d', 'Color', c_bcjr, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_bcjr, 'DisplayName', 'M-BCJR');
if ~isnan(ber_nb(t,1))
    semilogy(SNR_range, ber_nb(t,:), '-o', 'Color', c_nb, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_nb, 'DisplayName', 'Neighbors+DF');
end
if ~isnan(ber_win(t,1))
    semilogy(SNR_range, ber_win(t,:), '-s', 'Color', c_win, 'LineWidth', 2.5, 'MarkerSize', 11, 'MarkerFaceColor', c_win, 'DisplayName', 'Window+DF');
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('FTN Detection Performance (\\tau = %.1f)', tau_values(t)), 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/new/ber_tau06.png');
saveas(gcf, 'figures/new/ber_tau06.fig');
fprintf('Saved: figures/new/ber_tau06.png\n');

%% Figure 2: τ = 0.7
figure('Position', [150 100 700 550], 'Color', 'w');
t = 2;  % tau = 0.7

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th(t,:), '-^', 'Color', c_th, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_th, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss(t,:), '-v', 'Color', c_sss, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_sss, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_bcjr(t,:), '-d', 'Color', c_bcjr, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_bcjr, 'DisplayName', 'M-BCJR');
if ~isnan(ber_nb(t,1))
    semilogy(SNR_range, ber_nb(t,:), '-o', 'Color', c_nb, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_nb, 'DisplayName', 'Neighbors+DF');
end
if ~isnan(ber_win(t,1))
    semilogy(SNR_range, ber_win(t,:), '-s', 'Color', c_win, 'LineWidth', 2.5, 'MarkerSize', 11, 'MarkerFaceColor', c_win, 'DisplayName', 'Window+DF');
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('FTN Detection Performance (\\tau = %.1f)', tau_values(t)), 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/new/ber_tau07.png');
saveas(gcf, 'figures/new/ber_tau07.fig');
fprintf('Saved: figures/new/ber_tau07.png\n');

%% Figure 3: τ = 0.8
figure('Position', [200 100 700 550], 'Color', 'w');
t = 3;  % tau = 0.8

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th(t,:), '-^', 'Color', c_th, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_th, 'DisplayName', 'Threshold');
semilogy(SNR_range, ber_sss(t,:), '-v', 'Color', c_sss, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_sss, 'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_bcjr(t,:), '-d', 'Color', c_bcjr, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_bcjr, 'DisplayName', 'M-BCJR');
if ~isnan(ber_nb(t,1))
    semilogy(SNR_range, ber_nb(t,:), '-o', 'Color', c_nb, 'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', c_nb, 'DisplayName', 'Neighbors+DF');
end
if ~isnan(ber_win(t,1))
    semilogy(SNR_range, ber_win(t,:), '-s', 'Color', c_win, 'LineWidth', 2.5, 'MarkerSize', 11, 'MarkerFaceColor', c_win, 'DisplayName', 'Window+DF');
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('FTN Detection Performance (\\tau = %.1f)', tau_values(t)), 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/new/ber_tau08.png');
saveas(gcf, 'figures/new/ber_tau08.fig');
fprintf('Saved: figures/new/ber_tau08.png\n');

%% Figure 4: Window+DF All τ Comparison
figure('Position', [250 100 700 550], 'Color', 'w');

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;

tau_colors = {[0 0 0.9], [0.9 0 0], [0 0.7 0]};
markers = {'s', 'o', '^'};

for t = 1:length(tau_values)
    if ~isnan(ber_win(t,1))
        semilogy(SNR_range, ber_win(t,:), ['-' markers{t}], 'Color', tau_colors{t}, ...
            'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', tau_colors{t}, ...
            'DisplayName', sprintf('\\tau = %.1f', tau_values(t)));
    end
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF Performance vs Packing Factor', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/new/ber_window_all_tau.png');
saveas(gcf, 'figures/new/ber_window_all_tau.fig');
fprintf('Saved: figures/new/ber_window_all_tau.png\n');

%% Figure 5: Bar Chart @ 10 dB
figure('Position', [300 100 800 500], 'Color', 'w');

idx10 = find(SNR_range == 10);
data_10dB = [ber_th(:,idx10), ber_sss(:,idx10), ber_bcjr(:,idx10), ber_nb(:,idx10), ber_win(:,idx10)];
detector_names = {'Threshold', 'SSSgbKSE', 'M-BCJR', 'Neighbors+DF', 'Window+DF'};

b = bar(data_10dB);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('\\tau = %.1f', x), tau_values, 'UniformOutput', false));
set(gca, 'YScale', 'log');

bar_colors = [c_th; c_sss; c_bcjr; c_nb; c_win];
for k = 1:length(b)
    b(k).FaceColor = bar_colors(k,:);
end

ylabel('BER @ 10 dB', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Packing Factor (\tau)', 'FontSize', 14, 'FontWeight', 'bold');
title('Detection Performance Comparison @ E_b/N_0 = 10 dB', 'FontSize', 16, 'FontWeight', 'bold');
legend(detector_names, 'Location', 'northwest', 'FontSize', 10);
grid on;
ylim([1e-4 1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/new/ber_bar_10dB.png');
saveas(gcf, 'figures/new/ber_bar_10dB.fig');
fprintf('Saved: figures/new/ber_bar_10dB.png\n');

%% Figure 6: Gain Analysis - Window vs SSSgbKSE
figure('Position', [350 100 600 450], 'Color', 'w');

gain_vs_sss = ber_sss(:,idx10) ./ ber_win(:,idx10);

bar(tau_values, gain_vs_sss, 0.6, 'FaceColor', c_win);
xlabel('\tau', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Gain (×)', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF Gain over SSSgbKSE @ 10 dB', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

for i = 1:length(tau_values)
    text(tau_values(i), gain_vs_sss(i)+0.5, sprintf('%.1f×', gain_vs_sss(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

saveas(gcf, 'figures/new/gain_vs_sss.png');
saveas(gcf, 'figures/new/gain_vs_sss.fig');
fprintf('Saved: figures/new/gain_vs_sss.png\n');

%% Figure 7: Gain Analysis - Window vs Neighbors
figure('Position', [400 100 600 450], 'Color', 'w');

gain_vs_nb = ber_nb(:,idx10) ./ ber_win(:,idx10);

bar(tau_values, gain_vs_nb, 0.6, 'FaceColor', c_nb);
xlabel('\tau', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Gain (×)', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF Gain over Neighbors+DF @ 10 dB', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

for i = 1:length(tau_values)
    text(tau_values(i), gain_vs_nb(i)+0.3, sprintf('%.1f×', gain_vs_nb(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

saveas(gcf, 'figures/new/gain_vs_neighbors.png');
saveas(gcf, 'figures/new/gain_vs_neighbors.fig');
fprintf('Saved: figures/new/gain_vs_neighbors.png\n');

%% Summary Table
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                      BER Summary @ 10 dB                               ║\n');
fprintf('╠═══════╦════════════╦════════════╦════════════╦════════════╦═══════════╣\n');
fprintf('║   τ   ║  Threshold ║  SSSgbKSE  ║   M-BCJR   ║ Neighb+DF  ║ Window+DF ║\n');
fprintf('╠═══════╬════════════╬════════════╬════════════╬════════════╬═══════════╣\n');

for t = 1:length(tau_values)
    fprintf('║  %.1f  ║  %.2e  ║  %.2e  ║  %.2e  ║  %.2e  ║ %.2e  ║\n', ...
        tau_values(t), ber_th(t,idx10), ber_sss(t,idx10), ber_bcjr(t,idx10), ...
        ber_nb(t,idx10), ber_win(t,idx10));
end
fprintf('╚═══════╩════════════╩════════════╩════════════╩════════════╩═══════════╝\n');

fprintf('\n=== All %d figures saved to figures/new/ folder ===\n', 7);