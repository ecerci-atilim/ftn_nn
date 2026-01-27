%% FTN Detection: Structured 2D CNN vs Neighbor 1D CNN
% Based on Tokluoğlu et al. approach - Domain-informed structured input
% Each row represents ISI contribution from a specific neighbor symbol

clear; clc; close all;

%% Parameters
sps = 10;
beta = 0.3;
span = 6;
tau_values = [0.5, 0.6, 0.7, 0.8, 0.9];
SNR_range = 0:2:14;
SNR_train = 10;

N_train = 100000;
N_block = 10000;
min_errors = 100;
max_symbols = 1e6;

% Network parameters
max_epochs = 30;
mini_batch = 512;

% Neighbor symbols to consider
n_neighbors = 7;  % -3 to +3
n_samples_per_symbol = 7;  % samples around each symbol instant

%% Create directories
[~,~] = mkdir('results');
[~,~] = mkdir('figures');

%% Generate pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

%% Print header
fprintf('================================================================\n');
fprintf('  FTN Detection: Structured 2D CNN vs Neighbor 1D CNN\n');
fprintf('================================================================\n');
fprintf('Tau values:       %s\n', num2str(tau_values));
fprintf('SNR range:        %d to %d dB\n', SNR_range(1), SNR_range(end));
fprintf('Training SNR:     %d dB\n', SNR_train);
fprintf('Structured input: %dx%d matrix\n', n_neighbors, n_samples_per_symbol);
fprintf('Neighbor input:   %d samples\n', n_neighbors);
fprintf('================================================================\n');

total_timer = tic;

%% Main loop over tau
all_results = struct();

for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    step = round(tau * sps);
    
    fprintf('\n================================================================\n');
    fprintf('  [%d/%d] tau = %.1f (T_ftn = %d samples)\n', ...
            tau_idx, length(tau_values), tau, step);
    fprintf('================================================================\n');
    
    %% Generate training data
    fprintf('\nGenerating training data (N=%d, SNR=%ddB)...\n', N_train, SNR_train);
    rng(42);
    bits_train = randi([0 1], N_train, 1);
    symbols_train = 2*bits_train - 1;
    
    [rx_train, sym_idx_train] = generate_ftn_rx(symbols_train, step, h, delay, SNR_train, sps);
    
    %% Extract features
    fprintf('Extracting features...\n');
    
    % Margins
    margin = 3 * step + ceil(n_samples_per_symbol/2) + 5;
    valid_idx = (margin+1):(N_train-margin);
    n_valid = length(valid_idx);
    
    % Neighbor features: 7x1 (symbol instants only)
    X_neighbor = zeros(n_neighbors, 1, 1, n_valid);
    
    % Structured features: 7x7 matrix (7 symbols × 7 samples each)
    X_structured = zeros(n_neighbors, n_samples_per_symbol, 1, n_valid);
    
    neighbor_offsets = (-3:3) * step;  % Symbol instant offsets
    sample_offsets = floor(-(n_samples_per_symbol-1)/2):floor((n_samples_per_symbol-1)/2);
    
    for i = 1:n_valid
        k = valid_idx(i);
        center = sym_idx_train(k);
        
        % Neighbor: just symbol instants
        X_neighbor(:, 1, 1, i) = rx_train(center + neighbor_offsets);
        
        % Structured: each row = samples around each neighbor symbol
        for row = 1:n_neighbors
            sym_center = center + neighbor_offsets(row);
            X_structured(row, :, 1, i) = rx_train(sym_center + sample_offsets);
        end
    end
    
    Y_train = categorical(bits_train(valid_idx));
    
    % Normalize
    mu_n = mean(X_neighbor, 'all'); sig_n = std(X_neighbor, 0, 'all');
    X_neighbor = (X_neighbor - mu_n) / sig_n;
    
    mu_s = mean(X_structured, 'all'); sig_s = std(X_structured, 0, 'all');
    X_structured = (X_structured - mu_s) / sig_s;
    
    %% Count parameters
    % Neighbor 1D CNN: Conv(3x1, 32) + Conv(3x1, 16) + FC(2)
    %   = 3*32 + 32*3*16 + 16*2 = 96 + 1536 + 32 = 1664 params (approx)
    % Structured 2D CNN: Conv(1x7, 32) + Conv(7x1, 16) + FC(2)
    %   = 7*32 + 32*7*16 + 16*2 = 224 + 3584 + 32 = 3840 params (approx)
    
    fprintf('\n--- Comparison Setup ---\n');
    fprintf('Neighbor 1D CNN:   %d input, ~1700 params\n', n_neighbors);
    fprintf('Structured 2D CNN: %dx%d input, ~3800 params\n', n_neighbors, n_samples_per_symbol);
    fprintf('------------------------\n');
    
    %% Define and train networks
    opts = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    % --- Neighbor 1D CNN ---
    fprintf('\n[1/2] Training Neighbor 1D CNN...\n');
    tic;
    
    layers_1d = [
        imageInputLayer([n_neighbors 1 1], 'Normalization', 'none', 'Name', 'input')
        
        convolution2dLayer([3 1], 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        convolution2dLayer([3 1], 16, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        
        fullyConnectedLayer(2, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    net_1d = trainNetwork(X_neighbor, Y_train, layers_1d, opts);
    fprintf('  Done (%.1fs)\n', toc);
    
    % --- Structured 2D CNN ---
    fprintf('\n[2/2] Training Structured 2D CNN...\n');
    tic;
    
    layers_2d = [
        imageInputLayer([n_neighbors n_samples_per_symbol 1], 'Normalization', 'none', 'Name', 'input')
        
        % Horizontal scan: process samples within each symbol's region
        convolution2dLayer([1 7], 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        % Vertical scan: combine information across symbols
        convolution2dLayer([7 1], 16, 'Padding', 0, 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        
        fullyConnectedLayer(2, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    net_2d = trainNetwork(X_structured, Y_train, layers_2d, opts);
    fprintf('  Done (%.1fs)\n', toc);
    
    %% Test both approaches
    results = struct();
    approaches = {'Neighbor_1DCNN', 'Structured_2DCNN'};
    
    for app_idx = 1:2
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
                
                % Valid range
                valid_test = (margin+1):(N_block-margin);
                n_test = length(valid_test);
                
                if app_idx == 1
                    % Neighbor 1D CNN
                    X_test = zeros(n_neighbors, 1, 1, n_test);
                    for i = 1:n_test
                        k = valid_test(i);
                        X_test(:, 1, 1, i) = rx_test(sym_idx_test(k) + neighbor_offsets);
                    end
                    X_test = (X_test - mu_n) / sig_n;
                    pred = classify(net_1d, X_test);
                else
                    % Structured 2D CNN
                    X_test = zeros(n_neighbors, n_samples_per_symbol, 1, n_test);
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        for row = 1:n_neighbors
                            sym_center = center + neighbor_offsets(row);
                            X_test(row, :, 1, i) = rx_test(sym_center + sample_offsets);
                        end
                    end
                    X_test = (X_test - mu_s) / sig_s;
                    pred = classify(net_2d, X_test);
                end
                
                bits_hat = double(pred) - 1;  % categorical '0'->1, '1'->2
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
    
    %% Save results
    save(sprintf('results/cnn_results_tau%.1f.mat', tau), 'results', 'SNR_range', 'tau');
    
    %% Plot
    fig = figure('Position', [100 100 900 600]);
    hold on;
    set(gca, 'YScale', 'log');
    legends = {};
    
    % Load reference if exists
    ref_file = sprintf('../final_test0001/results/reference_tau%.1f.mat', tau);
    if exist(ref_file, 'file')
        load(ref_file, 'results_ref');
        if isfield(results_ref, 'bcjr')
            semilogy(results_ref.bcjr.SNR, results_ref.bcjr.BER, 'k-^', ...
                'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'k');
            legends{end+1} = 'M-BCJR';
        end
        if isfield(results_ref, 'sss')
            semilogy(results_ref.sss.SNR, results_ref.sss.BER, 'k--v', ...
                'LineWidth', 1.5, 'MarkerSize', 6);
            legends{end+1} = 'SSSgbKSE';
        end
        if isfield(results_ref, 'threshold')
            semilogy(results_ref.threshold.SNR, results_ref.threshold.BER, 'k:', ...
                'LineWidth', 1.5);
            legends{end+1} = 'Threshold';
        end
    end
    
    % Plot CNN results
    colors = [0.8 0.2 0.2; 0.2 0.2 0.8];  % Red for Neighbor, Blue for Structured
    markers = {'o--', 's-'};
    
    semilogy(SNR_range, results.Neighbor_1DCNN.BER, markers{1}, ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 7);
    legends{end+1} = 'Neighbor 1D-CNN (7 samples)';
    
    semilogy(SNR_range, results.Structured_2DCNN.BER, markers{2}, ...
        'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', colors(2,:));
    legends{end+1} = 'Structured 2D-CNN (7×7 matrix)';
    
    grid on;
    xlabel('$E_b/N_0$ (dB)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('BER', 'Interpreter', 'latex', 'FontSize', 12);
    title(sprintf('Structured 2D-CNN vs Neighbor 1D-CNN: $\\tau = %.1f$', tau), ...
        'Interpreter', 'latex', 'FontSize', 14);
    legend(legends, 'Location', 'southwest', 'FontSize', 10);
    ylim([1e-5 1]);
    xlim([0 14]);
    
    saveas(fig, sprintf('figures/cnn_comparison_tau%.1f.fig', tau));
    print(fig, sprintf('figures/cnn_comparison_tau%.1f.png', tau), '-dpng', '-r150');
    fprintf('\nSaved: figures/cnn_comparison_tau%.1f.png\n', tau);
    
    % Store for summary
    all_results.(['tau' strrep(sprintf('%.1f', tau), '.', '_')]) = results;
end

%% Summary table
fprintf('\n================================================================\n');
fprintf('                    BER Summary @ 10 dB\n');
fprintf('================================================================\n');
fprintf('tau    | Neighbor 1D | Structured 2D | Improvement\n');
fprintf('-------|-------------|---------------|------------\n');

for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    field = ['tau' strrep(sprintf('%.1f', tau), '.', '_')];
    
    ber_1d = all_results.(field).Neighbor_1DCNN.BER(SNR_range == 10);
    ber_2d = all_results.(field).Structured_2DCNN.BER(SNR_range == 10);
    improvement = ber_1d / ber_2d;
    
    fprintf('%.1f    | %.2e    | %.2e      | %.1fx\n', tau, ber_1d, ber_2d, improvement);
end
fprintf('================================================================\n');

fprintf('\n  Total simulation time: %.1f minutes\n', toc(total_timer)/60);
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