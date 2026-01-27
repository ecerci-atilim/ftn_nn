%% Visualize Sample Selection for Each Approach
% Shows which samples are selected as NN input for each approach

clear; clc; close all;

%% Parameters
sps = 10;
beta = 0.3;
span = 6;
tau = 0.7;
T_ftn = round(tau * sps);  % = 7

%% Generate example signal
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

rng(42);
N = 15;
bits = randi([0 1], N, 1);
symbols = 2*bits - 1;

% Transmit
tx = conv(upsample(symbols, T_ftn), h);
rx = conv(tx, h);
rx = rx / std(rx);

% Symbol indices
symbol_indices = delay + 1 + (0:N-1) * T_ftn;

%% Compute offsets
offsets.neighbor = (-3:3) * T_ftn;
offsets.fractional = round((-3:3) * (T_ftn-1) / 3);
offsets.fractional(4) = 0;
t1 = round(T_ftn / 3);
t2 = round(2 * T_ftn / 3);
offsets.hybrid = [-T_ftn, -t2, -t1, 0, t1, t2, T_ftn];

%% Target symbol
target_k = 8;  % which symbol to show
center = symbol_indices(target_k);

%% Colors
col_signal = [0.3 0.3 0.8];
col_neighbor = [0.8 0.2 0.2];
col_frac = [0.2 0.7 0.2];
col_hybrid = [0.8 0.5 0.0];
col_df = [0.5 0.2 0.7];

%% Create figure with 4 subplots
fig = figure('Position', [50 50 1200 800]);
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');

titles = {'Approach 1: Neighbor (Symbol-Rate)', ...
          'Approach 2: Fractional (Inter-Symbol)', ...
          'Approach 3: Hybrid', ...
          'Decision Feedback (D=4)'};
colors = {col_neighbor, col_frac, col_hybrid, col_df};
offset_sets = {offsets.neighbor, offsets.fractional, offsets.hybrid, (-4:-1)*T_ftn};

for sp = 1:4
    subplot(2,2,sp);
    
    % Plot signal
    plot(1:length(rx), rx, '-', 'Color', col_signal, 'LineWidth', 1);
    hold on;
    
    % Plot all symbol instants
    for k = 1:N
        si = symbol_indices(k);
        if si <= length(rx)
            if k == target_k
                plot(si, rx(si), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
            else
                plot(si, rx(si), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
            end
        end
    end
    
    % Plot selected samples
    off = offset_sets{sp};
    for i = 1:length(off)
        idx = center + off(i);
        if idx >= 1 && idx <= length(rx)
            plot(idx, rx(idx), 's', 'Color', colors{sp}, 'MarkerSize', 12, ...
                 'MarkerFaceColor', colors{sp}, 'LineWidth', 2);
            % Draw vertical line
            plot([idx idx], [0 rx(idx)], '--', 'Color', colors{sp}, 'LineWidth', 1);
        end
    end
    
    % Highlight target region
    xl = [center + min(off) - 5, center + max(off) + 5];
    xlim(xl);
    ylim([-2.5 2.5]);
    
    xlabel('Sample Index', 'FontSize', 11);
    ylabel('Amplitude', 'FontSize', 11);
    title(titles{sp}, 'FontSize', 12);
    grid on;
    
    % Add offset info
    if sp < 4
        off_str = sprintf('Offsets: [%s]', num2str(off));
    else
        off_str = sprintf('Feedback from $\\hat{b}_{k-4}$ to $\\hat{b}_{k-1}$');
    end
    text(0.02, 0.98, off_str, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 9, 'BackgroundColor', 'w', 'Interpreter', 'latex');
end

sgtitle(sprintf('Sample Selection Comparison ($\\tau = %.1f$, $T_{ftn} = %d$ samples, Target: $k = %d$)', ...
        tau, T_ftn, target_k), 'FontSize', 14, 'Interpreter', 'latex');

%% Save
[~,~] = mkdir('figures');
saveas(fig, 'figures/sample_selection.fig');
print(fig, 'figures/sample_selection.eps', '-depsc2');
fprintf('Sample selection figure saved!\n');

%% Print offset summary
fprintf('\n=== Offset Summary (tau=%.1f, T_ftn=%d) ===\n', tau, T_ftn);
fprintf('Neighbor:    [%s] (symbol instants)\n', num2str(offsets.neighbor));
fprintf('Fractional:  [%s] (inter-symbol)\n', num2str(offsets.fractional));
fprintf('Hybrid:      [%s] (mixed)\n', num2str(offsets.hybrid));
fprintf('DF:          [%s] (previous decisions)\n', num2str((-4:-1)*T_ftn));