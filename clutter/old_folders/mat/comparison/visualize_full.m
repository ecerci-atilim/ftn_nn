%% FTN Results Visualization

clear; clc; close all;

%% Load Results
load('mat/comparison/full_results.mat');

fprintf('=== FTN Results Visualization ===\n');
fprintf('τ values: '); fprintf('%.1f ', tau_values); fprintf('\n');
fprintf('SNR range: %d to %d dB\n\n', SNR_range(1), SNR_range(end));

%% AWGN Theory
ber_theory = qfunc(sqrt(2 * 10.^(SNR_range/10)));

%% Color Scheme
colors = struct();
colors.awgn = [0 0 0];           % Black
colors.threshold = [0.8 0 0];     % Dark Red
colors.sss = [0 0.6 0];           % Dark Green
colors.bcjr = [0 0.7 0.7];        % Cyan
colors.neighbors = [0.8 0 0.8];   % Magenta
colors.window = [0 0 0.8];        % Dark Blue

%% Figure 1: Three Subplots (τ comparison)
figure('Position', [50 100 1400 450], 'Color', 'w');

for t = 1:length(tau_values)
    subplot(1, 3, t);
    
    % AWGN
    semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'AWGN Theory');
    hold on;
    
    % Threshold
    semilogy(SNR_range, ber_th(t,:), '-^', 'Color', colors.threshold, ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', colors.threshold, ...
        'DisplayName', 'Threshold');
    
    % SSSgbKSE
    semilogy(SNR_range, ber_sss(t,:), '-v', 'Color', colors.sss, ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', colors.sss, ...
        'DisplayName', 'SSSgbKSE');
    
    % M-BCJR
    semilogy(SNR_range, ber_bcjr(t,:), '-d', 'Color', colors.bcjr, ...
        'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', colors.bcjr, ...
        'DisplayName', 'M-BCJR');
    
    % Neighbors+DF
    if ~isnan(ber_nb(t,1))
        semilogy(SNR_range, ber_nb(t,:), '-o', 'Color', colors.neighbors, ...
            'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors.neighbors, ...
            'DisplayName', 'Neighbors+DF');
    end
    
    % Window+DF
    if ~isnan(ber_win(t,1))
        semilogy(SNR_range, ber_win(t,:), '-s', 'Color', colors.window, ...
            'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.window, ...
            'DisplayName', 'Window+DF');
    end
    
    grid on;
    xlabel('E_b/N_0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('BER', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('\\tau = %.1f', tau_values(t)), 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 9);
    ylim([1e-5 1]);
    xlim([SNR_range(1) SNR_range(end)]);
    set(gca, 'FontSize', 11, 'LineWidth', 1);
end

sgtitle('FTN Detection Performance Comparison', 'FontSize', 16, 'FontWeight', 'bold');

if ~exist('figures', 'dir'), mkdir('figures'); end
saveas(gcf, 'figures/ftn_comparison_3panel.png');
saveas(gcf, 'figures/ftn_comparison_3panel.fig');
fprintf('Saved: figures/ftn_comparison_3panel.png\n');

%% Figure 2: Single Combined Plot (τ=0.7 focus)
figure('Position', [100 100 800 600], 'Color', 'w');

t_idx = find(tau_values == 0.7);
if isempty(t_idx), t_idx = 2; end

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;
semilogy(SNR_range, ber_th(t_idx,:), '-^', 'Color', colors.threshold, ...
    'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', colors.threshold, ...
    'DisplayName', 'Threshold (Uncoded FTN)');
semilogy(SNR_range, ber_sss(t_idx,:), '-v', 'Color', colors.sss, ...
    'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', colors.sss, ...
    'DisplayName', 'SSSgbKSE');
semilogy(SNR_range, ber_bcjr(t_idx,:), '-d', 'Color', colors.bcjr, ...
    'LineWidth', 2, 'MarkerSize', 9, 'MarkerFaceColor', colors.bcjr, ...
    'DisplayName', 'M-BCJR');

if ~isnan(ber_nb(t_idx,1))
    semilogy(SNR_range, ber_nb(t_idx,:), '-o', 'Color', colors.neighbors, ...
        'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', colors.neighbors, ...
        'DisplayName', 'NN: Neighbors+DF');
end
if ~isnan(ber_win(t_idx,1))
    semilogy(SNR_range, ber_win(t_idx,:), '-s', 'Color', colors.window, ...
        'LineWidth', 2.5, 'MarkerSize', 12, 'MarkerFaceColor', colors.window, ...
        'DisplayName', 'NN: Window+DF');
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('FTN Detection Performance (\\tau = %.1f)', tau_values(t_idx)), ...
    'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/ftn_tau07_detailed.png');
saveas(gcf, 'figures/ftn_tau07_detailed.fig');
fprintf('Saved: figures/ftn_tau07_detailed.png\n');

%% Figure 3: Window+DF across all τ values
figure('Position', [150 100 800 600], 'Color', 'w');

semilogy(SNR_range, ber_theory, 'k--', 'LineWidth', 2.5, 'DisplayName', 'AWGN Theory');
hold on;

tau_colors = {[0 0 0.9], [0.9 0 0], [0 0.7 0]};  % Blue, Red, Green
markers = {'s', 'o', '^'};

for t = 1:length(tau_values)
    if ~isnan(ber_win(t,1))
        semilogy(SNR_range, ber_win(t,:), ['-' markers{t}], 'Color', tau_colors{t}, ...
            'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', tau_colors{t}, ...
            'DisplayName', sprintf('Window+DF (\\tau=%.1f)', tau_values(t)));
    end
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF Performance vs Packing Factor', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
ylim([1e-5 1]);
xlim([SNR_range(1) SNR_range(end)]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

saveas(gcf, 'figures/ftn_window_all_tau.png');
saveas(gcf, 'figures/ftn_window_all_tau.fig');
fprintf('Saved: figures/ftn_window_all_tau.png\n');

%% Figure 4: Bar Chart @ 10 dB
figure('Position', [200 100 900 500], 'Color', 'w');

idx10 = find(SNR_range == 10);
if isempty(idx10), idx10 = 6; end

n_tau = length(tau_values);
data_10dB = [ber_th(:,idx10), ber_sss(:,idx10), ber_bcjr(:,idx10), ber_nb(:,idx10), ber_win(:,idx10)];
detector_names = {'Threshold', 'SSSgbKSE', 'M-BCJR', 'Neighbors+DF', 'Window+DF'};

b = bar(data_10dB);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('\\tau=%.1f', x), tau_values, 'UniformOutput', false));
set(gca, 'YScale', 'log');

% Colors
bar_colors = [colors.threshold; colors.sss; colors.bcjr; colors.neighbors; colors.window];
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

saveas(gcf, 'figures/ftn_bar_10dB.png');
saveas(gcf, 'figures/ftn_bar_10dB.fig');
fprintf('Saved: figures/ftn_bar_10dB.png\n');

%% Figure 5: Gain Analysis (Window+DF vs others)
figure('Position', [250 100 800 500], 'Color', 'w');

gain_vs_sss = ber_sss ./ ber_win;
gain_vs_nb = ber_nb ./ ber_win;

subplot(1,2,1);
bar(tau_values, gain_vs_sss(:,idx10), 'FaceColor', colors.window);
xlabel('\tau', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Gain (×)', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF vs SSSgbKSE @ 10 dB', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);

subplot(1,2,2);
bar(tau_values, gain_vs_nb(:,idx10), 'FaceColor', colors.neighbors);
xlabel('\tau', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Gain (×)', 'FontSize', 14, 'FontWeight', 'bold');
title('Window+DF vs Neighbors+DF @ 10 dB', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);

sgtitle('Performance Gains of Window+DF Approach', 'FontSize', 15, 'FontWeight', 'bold');

% saveas(gcf, 'figures/ftn_gain_analysis.png');
% saveas(gcf, 'figures/ftn_gain_analysis.fig');
% fprintf('Saved: figures/ftn_gain_analysis.png\n');

%% Print Summary Table
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                      BER Summary @ 10 dB                               ║\n');
fprintf('╠═══════╦════════════╦════════════╦════════════╦════════════╦═══════════╣\n');
fprintf('║   τ   ║  Threshold ║  SSSgbKSE  ║   M-BCJR   ║ Neighb+DF  ║ Window+DF ║\n');
fprintf('╠═══════╬════════════╬════════════╬════════════╬════════════╬═══════════╣\n');

for t = 1:n_tau
    fprintf('║  %.1f  ║  %.2e  ║  %.2e  ║  %.2e  ║  %.2e  ║ %.2e  ║\n', ...
        tau_values(t), ber_th(t,idx10), ber_sss(t,idx10), ber_bcjr(t,idx10), ...
        ber_nb(t,idx10), ber_win(t,idx10));
end
fprintf('╚═══════╩════════════╩════════════╩════════════╩════════════╩═══════════╝\n');

fprintf('\n--- Gains @ 10 dB ---\n');
for t = 1:n_tau
    fprintf('τ=%.1f: Window+DF is %.1fx better than SSSgbKSE, %.1fx better than Neighbors+DF\n', ...
        tau_values(t), gain_vs_sss(t,idx10), gain_vs_nb(t,idx10));
end

fprintf('\n=== All figures saved to figures/ folder ===\n');