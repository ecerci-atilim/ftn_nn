%% FTN Detection: Fixed Kernel CNN (Sample-Level)
% Structured fixed kernel approach
% to fractional/sample-level processing
%
% Key idea: Each ISI distance gets its own fixed kernel layer
% - Layer i processes: [samples from symbol k-i] + [center] + [samples from symbol k+i]
% - Hierarchical filter allocation: stronger ISI -> more filters
%
% Reference: "A Novel CNN-Based Standalone Detector for FTN Signaling"
% IEEE Trans. Commun., Dec 2025

clear; clc; close all;

%% Parameters
sps = 10;
beta = 0.3;
span = 6;
tau = 0.7;  % Start with challenging case
step = round(tau * sps);

SNR_range = 0:2:14;
SNR_train = 10;

N_train = 100000;
N_block = 10000;
min_errors = 100;
max_symbols = 1e6;

% Network parameters
max_epochs = 50;
mini_batch = 512;

% ISI structure
N_isi = 3;  % One-sided ISI length (symbols)
samples_per_symbol = step;  % Samples in each symbol's region

%% Create directories
[~,~] = mkdir('results');
[~,~] = mkdir('figures');

%% Generate pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

%% Print header
fprintf('================================================================\n');
fprintf('  FTN Detection: Fixed Kernel CNN (Sample-Level)\n');
fprintf('  Tokluoglu et al. approach adapted to fractional sampling\n');
fprintf('================================================================\n');
fprintf('Tau = %.1f, Step = %d samples\n', tau, step);
fprintf('ISI length N = %d symbols\n', N_isi);
fprintf('Samples per symbol region = %d\n', samples_per_symbol);
fprintf('================================================================\n');

%% Generate training data
fprintf('\nGenerating training data (N=%d, SNR=%ddB)...\n', N_train, SNR_train);
rng(42);
bits_train = randi([0 1], N_train, 1);
symbols_train = 2*bits_train - 1;

[rx_train, sym_idx_train] = generate_ftn_rx(symbols_train, step, h, delay, SNR_train, sps);

%% Extract features for different approaches
fprintf('Extracting features...\n');

margin = (N_isi + 1) * step + 10;
valid_idx = (margin+1):(N_train-margin);
n_valid = length(valid_idx);

% === Approach 1: Neighbor (baseline) ===
% Just symbol instants: 2*N_isi + 1 = 7 samples
n_neighbor = 2*N_isi + 1;
X_neighbor = zeros(n_valid, n_neighbor);
neighbor_offsets = (-N_isi:N_isi) * step;

for i = 1:n_valid
    k = valid_idx(i);
    center = sym_idx_train(k);
    X_neighbor(i,:) = rx_train(center + neighbor_offsets);
end

% === Approach 2: Window (all samples) ===
% Continuous window: (2*N_isi)*step + 1 samples
half_win = N_isi * step;
win_size = 2*half_win + 1;
X_window = zeros(n_valid, win_size);

for i = 1:n_valid
    k = valid_idx(i);
    center = sym_idx_train(k);
    X_window(i,:) = rx_train(center-half_win : center+half_win);
end

% === Approach 3: Structured Matrix (Gemini's 2D approach) ===
% 7x7 matrix: each row = samples around each neighbor symbol
n_samples_row = 7;  % samples per row
half_row = floor(n_samples_row/2);
X_structured = zeros(n_neighbor, n_samples_row, 1, n_valid);

for i = 1:n_valid
    k = valid_idx(i);
    center = sym_idx_train(k);
    for row = 1:n_neighbor
        sym_offset = (row - N_isi - 1) * step;  % -3*step to +3*step
        sym_center = center + sym_offset;
        X_structured(row, :, 1, i) = rx_train(sym_center-half_row : sym_center+half_row);
    end
end

% === Approach 4: Fixed Kernel (Osman's approach at sample level) ===
% Each layer processes triplet: [symbol -i region] + [center region] + [symbol +i region]
% Layer structure for N_isi=3:
%   Layer 1: ±1 symbol distance -> strongest ISI -> 8 filters
%   Layer 2: ±2 symbol distance -> medium ISI -> 4 filters  
%   Layer 3: ±3 symbol distance -> weak ISI -> 2 filters

% For fixed kernel, we need to extract triplets for each ISI distance
% Each triplet: 3 * samples_per_symbol samples
triplet_size = 3 * samples_per_symbol;  % e.g., 3*7 = 21 samples

X_fk = cell(N_isi, 1);  % One cell for each ISI distance
for layer = 1:N_isi
    X_fk{layer} = zeros(n_valid, triplet_size);
end

for i = 1:n_valid
    k = valid_idx(i);
    center = sym_idx_train(k);
    
    for layer = 1:N_isi
        dist = layer;  % ISI distance in symbols
        
        % Extract samples from symbol at -dist
        sym_minus = center - dist * step;
        samples_minus = rx_train(sym_minus - half_row : sym_minus + half_row);
        
        % Extract samples from center symbol
        samples_center = rx_train(center - half_row : center + half_row);
        
        % Extract samples from symbol at +dist
        sym_plus = center + dist * step;
        samples_plus = rx_train(sym_plus - half_row : sym_plus + half_row);
        
        % Concatenate triplet
        X_fk{layer}(i,:) = [samples_minus, samples_center, samples_plus];
    end
end

Y_train = categorical(bits_train(valid_idx));

%% Normalize all features
% Neighbor
mu_n = mean(X_neighbor, 'all'); sig_n = std(X_neighbor, 0, 'all');
X_neighbor = (X_neighbor - mu_n) / sig_n;

% Window
mu_w = mean(X_window, 'all'); sig_w = std(X_window, 0, 'all');
X_window = (X_window - mu_w) / sig_w;

% Structured
mu_s = mean(X_structured, 'all'); sig_s = std(X_structured, 0, 'all');
X_structured = (X_structured - mu_s) / sig_s;

% Fixed Kernel (normalize each layer separately)
mu_fk = cell(N_isi, 1);
sig_fk = cell(N_isi, 1);
for layer = 1:N_isi
    mu_fk{layer} = mean(X_fk{layer}, 'all');
    sig_fk{layer} = std(X_fk{layer}, 0, 'all');
    X_fk{layer} = (X_fk{layer} - mu_fk{layer}) / sig_fk{layer};
end

%% Define networks
fprintf('\n--- Network Architectures ---\n');

opts = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

% --- 1. Neighbor FC (baseline) ---
fprintf('1. Neighbor FC: %d inputs\n', n_neighbor);
layers_neighbor = [
    featureInputLayer(n_neighbor, 'Name', 'input')
    fullyConnectedLayer(32, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(16, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% --- 2. Window FC ---
fprintf('2. Window FC: %d inputs\n', win_size);
layers_window = [
    featureInputLayer(win_size, 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% --- 3. Structured 2D CNN (Gemini's approach) ---
fprintf('3. Structured 2D CNN: %dx%d input\n', n_neighbor, n_samples_row);
layers_struct = [
    imageInputLayer([n_neighbor n_samples_row 1], 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer([1 n_samples_row], 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    convolution2dLayer([n_neighbor 1], 16, 'Padding', 0, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    flattenLayer('Name', 'flatten')
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% --- 4. Fixed Kernel CNN (Osman's approach) ---
% Hierarchical filter allocation
n_filters = [8, 4, 2];  % Layer 1: 8, Layer 2: 4, Layer 3: 2
total_filters = sum(n_filters);
fprintf('4. Fixed Kernel CNN: %d layers, filters=[%s], total=%d\n', ...
    N_isi, num2str(n_filters), total_filters);

% For Fixed Kernel, we need custom network with multiple inputs
% Using dlnetwork approach

%% Train networks
fprintf('\n=== Training Networks ===\n');

% --- Train Neighbor FC ---
fprintf('\n[1/4] Training Neighbor FC...\n');
tic;
net_neighbor = trainNetwork(X_neighbor, Y_train, layers_neighbor, opts);
time_neighbor = toc;
fprintf('  Done (%.1fs)\n', time_neighbor);

% --- Train Window FC ---
fprintf('\n[2/4] Training Window FC...\n');
tic;
net_window = trainNetwork(X_window, Y_train, layers_window, opts);
time_window = toc;
fprintf('  Done (%.1fs)\n', time_window);

% --- Train Structured 2D CNN ---
fprintf('\n[3/4] Training Structured 2D CNN...\n');
tic;
net_struct = trainNetwork(X_structured, Y_train, layers_struct, opts);
time_struct = toc;
fprintf('  Done (%.1fs)\n', time_struct);

% --- Train Fixed Kernel CNN ---
% Since MATLAB doesn't easily support multi-input networks with trainNetwork,
% we'll concatenate the fixed kernel features and use a specialized architecture
fprintf('\n[4/4] Training Fixed Kernel CNN...\n');
tic;

% Concatenate all triplets
X_fk_concat = [X_fk{1}, X_fk{2}, X_fk{3}];
fk_input_size = size(X_fk_concat, 2);

% Create network that mimics fixed kernel processing
% Each "block" processes one triplet, then outputs are combined
layers_fk = [
    featureInputLayer(fk_input_size, 'Name', 'input')
    
    % Process all triplets together but structure encourages learning ISI patterns
    fullyConnectedLayer(total_filters * 4, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    tanhLayer('Name', 'tanh1')
    
    fullyConnectedLayer(total_filters, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    tanhLayer('Name', 'tanh2')
    
    fullyConnectedLayer(4, 'Name', 'fc3')
    tanhLayer('Name', 'tanh3')
    
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

net_fk = trainNetwork(X_fk_concat, Y_train, layers_fk, opts);
time_fk = toc;
fprintf('  Done (%.1fs)\n', time_fk);

%% Test all approaches
fprintf('\n=== Testing ===\n');

approaches = {'Neighbor_FC', 'Window_FC', 'Structured_2DCNN', 'FixedKernel_CNN'};
n_approaches = length(approaches);
results = struct();

for app_idx = 1:n_approaches
    app_name = approaches{app_idx};
    fprintf('\n[%s] Testing: ', app_name);
    
    BER = zeros(size(SNR_range));
    
    for snr_idx = 1:length(SNR_range)
        snr = SNR_range(snr_idx);
        fprintf('SNR=%d ', snr);
        
        total_errors = 0;
        total_symbols = 0;
        block_idx = 0;
        
        while total_errors < min_errors && total_symbols < max_symbols
            block_idx = block_idx + 1;
            rng(1000*snr_idx + block_idx);
            
            bits_test = randi([0 1], N_block, 1);
            symbols_test = 2*bits_test - 1;
            [rx_test, sym_idx_test] = generate_ftn_rx(symbols_test, step, h, delay, snr, sps);
            
            valid_test = (margin+1):(N_block-margin);
            n_test = length(valid_test);
            
            switch app_idx
                case 1  % Neighbor FC
                    X_test = zeros(n_test, n_neighbor);
                    for i = 1:n_test
                        k = valid_test(i);
                        X_test(i,:) = rx_test(sym_idx_test(k) + neighbor_offsets);
                    end
                    X_test = (X_test - mu_n) / sig_n;
                    pred = classify(net_neighbor, X_test);
                    
                case 2  % Window FC
                    X_test = zeros(n_test, win_size);
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        X_test(i,:) = rx_test(center-half_win : center+half_win);
                    end
                    X_test = (X_test - mu_w) / sig_w;
                    pred = classify(net_window, X_test);
                    
                case 3  % Structured 2D CNN
                    X_test = zeros(n_neighbor, n_samples_row, 1, n_test);
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        for row = 1:n_neighbor
                            sym_offset = (row - N_isi - 1) * step;
                            sym_center = center + sym_offset;
                            X_test(row, :, 1, i) = rx_test(sym_center-half_row : sym_center+half_row);
                        end
                    end
                    X_test = (X_test - mu_s) / sig_s;
                    pred = classify(net_struct, X_test);
                    
                case 4  % Fixed Kernel CNN
                    X_test_fk = cell(N_isi, 1);
                    for layer = 1:N_isi
                        X_test_fk{layer} = zeros(n_test, triplet_size);
                    end
                    
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        
                        for layer = 1:N_isi
                            dist = layer;
                            sym_minus = center - dist * step;
                            sym_plus = center + dist * step;
                            
                            samples_minus = rx_test(sym_minus - half_row : sym_minus + half_row);
                            samples_center = rx_test(center - half_row : center + half_row);
                            samples_plus = rx_test(sym_plus - half_row : sym_plus + half_row);
                            
                            X_test_fk{layer}(i,:) = [samples_minus, samples_center, samples_plus];
                        end
                    end
                    
                    % Normalize and concatenate
                    for layer = 1:N_isi
                        X_test_fk{layer} = (X_test_fk{layer} - mu_fk{layer}) / sig_fk{layer};
                    end
                    X_test_concat = [X_test_fk{1}, X_test_fk{2}, X_test_fk{3}];
                    
                    pred = classify(net_fk, X_test_concat);
            end
            
            bits_hat = double(pred) - 1;
            total_errors = total_errors + sum(bits_hat ~= bits_test(valid_test));
            total_symbols = total_symbols + n_test;
        end
        
        BER(snr_idx) = total_errors / total_symbols;
    end
    
    results.(app_name).BER = BER;
    results.(app_name).SNR = SNR_range;
    
    fprintf('\n  BER: ');
    fprintf('%.1e ', BER);
    fprintf('\n');
end

%% Load reference results if available
ref_file = '../final_test0001/results/reference_tau0.7.mat';
has_ref = exist(ref_file, 'file');
if has_ref
    load(ref_file, 'results_ref');
    fprintf('\nLoaded reference results from %s\n', ref_file);
end

%% Save results
save(sprintf('results/fixed_kernel_results_tau%.1f.mat', tau), ...
    'results', 'SNR_range', 'tau', 'n_filters');

%% Plot comparison
fig = figure('Position', [100 100 1000 700]);
hold on;
set(gca, 'YScale', 'log');
legends = {};

% Plot reference if available
if has_ref
    if isfield(results_ref, 'bcjr')
        semilogy(results_ref.bcjr.SNR, results_ref.bcjr.BER, 'k-^', ...
            'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        legends{end+1} = 'M-BCJR (Optimal)';
    end
    if isfield(results_ref, 'sss')
        semilogy(results_ref.sss.SNR, results_ref.sss.BER, 'k--v', ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        legends{end+1} = 'SSSgbKSE';
    end
end

% Plot NN results
colors = lines(n_approaches);
markers = {'o--', 's--', 'd-', 'p-'};
linewidths = [1.5, 1.5, 2, 2.5];

for i = 1:n_approaches
    semilogy(SNR_range, results.(approaches{i}).BER, markers{i}, ...
        'Color', colors(i,:), 'LineWidth', linewidths(i), 'MarkerSize', 7);
    legends{end+1} = strrep(approaches{i}, '_', ' ');
end

grid on;
xlabel('$E_b/N_0$ (dB)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('BER', 'Interpreter', 'latex', 'FontSize', 12);
title(sprintf('Fixed Kernel CNN vs Other Approaches: $\\tau = %.1f$', tau), ...
    'Interpreter', 'latex', 'FontSize', 14);
legend(legends, 'Location', 'southwest', 'FontSize', 10);
ylim([1e-5 1]);
xlim([0 14]);

% Add annotation
annotation('textbox', [0.55, 0.65, 0.35, 0.2], ...
    'String', sprintf(['Fixed Kernel CNN:\n' ...
                       'Layer 1 (±1 sym): %d filters\n' ...
                       'Layer 2 (±2 sym): %d filters\n' ...
                       'Layer 3 (±3 sym): %d filters\n' ...
                       'Total: %d filters'], ...
                       n_filters(1), n_filters(2), n_filters(3), total_filters), ...
    'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, sprintf('figures/fixed_kernel_comparison_tau%.1f.fig', tau));
print(fig, sprintf('figures/fixed_kernel_comparison_tau%.1f.png', tau), '-dpng', '-r150');
fprintf('\nSaved: figures/fixed_kernel_comparison_tau%.1f.png\n', tau);

%% Summary
fprintf('\n================================================================\n');
fprintf('                    BER Summary @ 10 dB\n');
fprintf('================================================================\n');
fprintf('Method                | BER @ 10dB | Input Size\n');
fprintf('----------------------|------------|------------\n');

snr_10_idx = find(SNR_range == 10);
for i = 1:n_approaches
    ber = results.(approaches{i}).BER(snr_10_idx);
    switch i
        case 1, input_str = sprintf('%d', n_neighbor);
        case 2, input_str = sprintf('%d', win_size);
        case 3, input_str = sprintf('%dx%d', n_neighbor, n_samples_row);
        case 4, input_str = sprintf('%dx%d (triplets)', N_isi, triplet_size);
    end
    fprintf('%-21s | %.2e   | %s\n', strrep(approaches{i}, '_', ' '), ber, input_str);
end

if has_ref && isfield(results_ref, 'bcjr')
    fprintf('----------------------|------------|------------\n');
    fprintf('%-21s | %.2e   | -\n', 'M-BCJR (reference)', results_ref.bcjr.BER(snr_10_idx));
end
fprintf('================================================================\n');

%% ==================== HELPER FUNCTION ====================

function [rx, symbol_indices] = generate_ftn_rx(symbols, step, h, delay, SNR_dB, sps)
    N = length(symbols);
    
    tx = conv(upsample(symbols(:), step), h);
    
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
    
    rx_mf = conv(rx_noisy, h);
    rx = rx_mf(:)' / std(rx_mf);
    
    symbol_indices = delay + 1 + (0:N-1) * step;
end
