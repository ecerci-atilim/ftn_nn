%% FTN Detection: Window vs Neighbor Comparison
% Compare Window (FC & BiLSTM) vs Neighbor (FC) - NO Decision Feedback

clear; clc; close all;

%% Parameters
sps = 10;
beta = 0.3;
span = 6;
tau_values = [0.7];  % Start with one
SNR_range = 0:2:14;

N_train = 50000;
N_block = 10000;
min_errors = 100;
max_symbols = 1e6;

% Training parameters
max_epochs = 50;
mini_batch = 512;

%% Create directories
[~,~] = mkdir('results');
[~,~] = mkdir('figures');

%% Generate pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

%% Print header
fprintf('================================================\n');
fprintf('  FTN Detection: Window vs Neighbor (No DF)\n');
fprintf('================================================\n');
fprintf('Tau values:    %s\n', num2str(tau_values));
fprintf('SNR range:     %d to %d dB\n', SNR_range(1), SNR_range(end));
fprintf('Training SNR:  8 dB\n');
fprintf('================================================\n');

total_timer = tic;

%% Main loop
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    T_ftn = round(tau * sps);
    
    fprintf('\n================================================\n');
    fprintf('  tau = %.1f (T_ftn = %d samples)\n', tau, T_ftn);
    fprintf('================================================\n');
    
    % Dimensions
    half_win = 3 * T_ftn;
    win_size = 2 * half_win + 1;
    neighbor_offsets = (-3:3) * T_ftn;
    n_neighbor = length(neighbor_offsets);
    
    fprintf('Window size: %d samples\n', win_size);
    fprintf('Neighbor samples: %d\n', n_neighbor);
    
    %% Generate training data
    fprintf('\nGenerating training data (N=%d)...\n', N_train);
    rng(42);
    bits_train = randi([0 1], 1, N_train);
    SNR_train = 8;
    [rx_train, sym_idx_train] = generate_ftn_rx(bits_train, tau, sps, h, delay, SNR_train);
    
    %% Extract features
    margin = half_win + 5;
    valid_idx = (margin+1):(N_train-margin);
    n_valid = length(valid_idx);
    
    % Neighbor features
    X_neighbor = zeros(n_valid, n_neighbor);
    % Window features (for GRU)
    X_window = zeros(n_valid, win_size);
    y_train = zeros(n_valid, 1);
    
    for i = 1:n_valid
        k = valid_idx(i);
        center = sym_idx_train(k);
        
        X_neighbor(i, :) = rx_train(center + neighbor_offsets);
        X_window(i, :) = rx_train(center-half_win : center+half_win);
        y_train(i) = bits_train(k);
    end
    
    y_cat = categorical(y_train);
    
    %% Count parameters for fair comparison
    % FC [32,16]: 7*32 + 32*16 + 16*2 = 768 params
    % BiLSTM(10): 8*10*12 = 960 params (close enough)
    % Window FC [64,32]: 43*64 + 64*32 + 32*2 = 4928 params (not fair, but test)
    
    fprintf('\n--- Parameter Count ---\n');
    fprintf('Neighbor_FC [32,16]:   %d params\n', n_neighbor*32 + 32*16 + 16*2);
    fprintf('Window_BiLSTM [10]:    ~960 params\n');
    fprintf('Window_FC [64,32]:     %d params (unfair but test)\n', win_size*64 + 64*32 + 32*2);
    fprintf('-----------------------\n');
    
    %% Train Neighbor_FC
    fprintf('\n[1/3] Training Neighbor_FC...\n');
    tic;
    
    layers_fc = [
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
    
    options_fc = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net_neighbor_fc = trainNetwork(X_neighbor, y_cat, layers_fc, options_fc);
    fprintf('  Done (%.1fs)\n', toc);
    
    %% Train Window_BiLSTM
    fprintf('\n[2/3] Training Window_BiLSTM...\n');
    tic;
    
    % Prepare sequence data: each row -> cell with 1 x win_size sequence
    X_seq = cell(n_valid, 1);
    for i = 1:n_valid
        X_seq{i} = X_window(i, :);  % 1 x win_size
    end
    
    layers_bilstm = [
        sequenceInputLayer(1, 'Name', 'input')
        bilstmLayer(10, 'OutputMode', 'last', 'Name', 'bilstm')
        fullyConnectedLayer(2, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    options_bilstm = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    net_bilstm = trainNetwork(X_seq, y_cat, layers_bilstm, options_bilstm);
    fprintf('  Done (%.1fs)\n', toc);
    
    %% Train Window_FC
    fprintf('\n[3/3] Training Window_FC...\n');
    tic;
    
    layers_window_fc = [
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
    
    net_window_fc = trainNetwork(X_window, y_cat, layers_window_fc, options_fc);
    fprintf('  Done (%.1fs)\n', toc);
    
    %% Test all approaches
    results = struct();
    approaches = {'Neighbor_FC', 'Window_BiLSTM', 'Window_FC'};
    
    for app_idx = 1:3
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
                
                bits_test = randi([0 1], 1, N_block);
                [rx_test, sym_idx_test] = generate_ftn_rx(bits_test, tau, sps, h, delay, snr);
                
                % Valid range
                valid_test = (margin+1):(N_block-margin);
                n_test = length(valid_test);
                
                if app_idx == 1
                    % Neighbor_FC - batch predict
                    X_test = zeros(n_test, n_neighbor);
                    for i = 1:n_test
                        k = valid_test(i);
                        X_test(i,:) = rx_test(sym_idx_test(k) + neighbor_offsets);
                    end
                    prob = predict(net_neighbor_fc, X_test);
                    bits_hat = (prob(:,2) > 0.5)';
                elseif app_idx == 2
                    % Window_BiLSTM - batch predict
                    X_test_seq = cell(n_test, 1);
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        X_test_seq{i} = rx_test(center-half_win : center+half_win);
                    end
                    prob = predict(net_bilstm, X_test_seq);
                    bits_hat = (prob(:,2) > 0.5)';
                else
                    % Window_FC - batch predict
                    X_test = zeros(n_test, win_size);
                    for i = 1:n_test
                        k = valid_test(i);
                        center = sym_idx_test(k);
                        X_test(i,:) = rx_test(center-half_win : center+half_win);
                    end
                    prob = predict(net_window_fc, X_test);
                    bits_hat = (prob(:,2) > 0.5)';
                end
                
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
    
    %% Save
    save(sprintf('results/results_tau%.1f.mat', tau), 'results', 'SNR_range', 'tau');
    
    %% Plot
    fig = figure('Position', [100 100 800 500]);
    hold on;
    set(gca, 'YScale', 'log');
    legends = {};
    
    % Load reference
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
    end
    
    % Plot NN results
    colors = lines(3);
    markers = {'o-', 's-', 'd-'};
    for i = 1:3
        semilogy(SNR_range, results.(approaches{i}).BER, markers{i}, ...
            'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 6);
        legends{end+1} = strrep(approaches{i}, '_', ' ');
    end
    
    grid on;
    xlabel('$E_b/N_0$ (dB)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('BER', 'Interpreter', 'latex', 'FontSize', 12);
    title(sprintf('Window vs Neighbor: $\\tau = %.1f$ (No DF)', tau), 'Interpreter', 'latex', 'FontSize', 13);
    legend(legends, 'Location', 'southwest', 'Interpreter', 'latex');
    ylim([1e-5 1]);
    xlim([0 14]);
    
    saveas(fig, sprintf('figures/ber_tau%.1f.fig', tau));
    print(fig, sprintf('figures/ber_tau%.1f.png', tau), '-dpng', '-r150');
    fprintf('\nSaved: figures/ber_tau%.1f.png\n', tau);
end

fprintf('\n================================================\n');
fprintf('  Complete! Total time: %.1f min\n', toc(total_timer)/60);
fprintf('================================================\n');

%% ==================== HELPER FUNCTION ====================

function [rx, symbol_indices] = generate_ftn_rx(bits, tau, sps, h, delay, SNR_dB)
    symbols = 2*bits - 1;
    step = round(tau * sps);
    N = length(bits);
    
    tx = conv(upsample(symbols(:), step), h);
    
    EbN0_lin = 10^(SNR_dB/10);
    noise_var = 1 / (2 * EbN0_lin);
    rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
    
    rx_mf = conv(rx_noisy, h);
    rx = rx_mf(:)' / std(rx_mf);
    
    symbol_indices = delay + 1 + (0:N-1) * step;
end