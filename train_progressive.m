% Progressive ISI Training: Start easy, get progressively harder
clear, clc, close all

%% Stage 1: Train with τ=0.95 (minimal ISI)
fprintf('=== STAGE 1: Training with τ=0.95 (minimal ISI) ===\n');
tau_stage1 = 0.95;
model_stage1 = train_universal_model(tau_stage1, 'stage1');

%% Stage 2: Fine-tune with τ=0.85 (moderate ISI)  
fprintf('\n=== STAGE 2: Fine-tuning with τ=0.85 (moderate ISI) ===\n');
tau_stage2 = 0.85;
model_stage2 = train_universal_model(tau_stage2, 'stage2', model_stage1);

%% Stage 3: Fine-tune with τ=0.7 (severe ISI)
fprintf('\n=== STAGE 3: Fine-tuning with τ=0.7 (target ISI) ===\n');
tau_final = 0.7;
final_model = train_universal_model(tau_final, 'final', model_stage2);

fprintf('\n=== PROGRESSIVE TRAINING COMPLETE ===\n');

function model = train_universal_model(tau_val, stage_name, pretrained_model)
    % Network parameters (same as before)
    modulation_type = 'psk'; modulation_order = 4; phase_offset = pi/4;
    window_len = 31; num_past_taps = 4; num_future_taps = 2; k = log2(modulation_order);
    signal_input_len = window_len * 2; num_history_features = (num_past_taps + num_future_taps) * 2;
    total_input_len = signal_input_len + num_history_features;
    
    % Generate training data with current τ
    fprintf('Generating training data for τ=%.2f...\n', tau_val);
    N_train = 100000; % More data for harder channels
    SNR_train_dB = 25;
    [x_train, y_train_bits] = generate_universal_data(N_train, tau_val, SNR_train_dB, modulation_type, modulation_order, phase_offset, window_len, num_past_taps, num_future_taps);
    
    % Data split
    cv = cvpartition(size(x_train, 1), 'HoldOut', 0.15);
    X_train_final = x_train(cv.training, :);
    Y_train_final = y_train_bits(cv.training, :);
    X_val = x_train(cv.test, :);
    Y_val = y_train_bits(cv.test, :);
    
    % Create or load network
    if exist('pretrained_model', 'var') && ~isempty(pretrained_model)
        fprintf('Fine-tuning from previous stage...\n');
        lgraph = layerGraph(pretrained_model);
        initial_lr = 0.0003; % Lower learning rate for fine-tuning
    else
        fprintf('Training from scratch...\n');
        % Create fresh network
        lgraph = layerGraph();
        input_layer = featureInputLayer(total_input_len, 'Name', 'combined_input', 'Normalization', 'zscore');
        lgraph = addLayers(lgraph, input_layer);
        
        % Larger, more powerful network for severe ISI
        deep_layers = [
            fullyConnectedLayer(512, 'Name', 'fc1')  % Increased capacity
            reluLayer('Name', 'relu1')
            batchNormalizationLayer('Name', 'bn1')
            
            fullyConnectedLayer(256, 'Name', 'fc2')
            reluLayer('Name', 'relu2')
            batchNormalizationLayer('Name', 'bn2')
            dropoutLayer(0.3, 'Name', 'dropout1')
            
            fullyConnectedLayer(128, 'Name', 'fc3')
            reluLayer('Name', 'relu3')
            batchNormalizationLayer('Name', 'bn3')
            dropoutLayer(0.3, 'Name', 'dropout2')
            
            fullyConnectedLayer(64, 'Name', 'fc4')
            reluLayer('Name', 'relu4')
            dropoutLayer(0.4, 'Name', 'dropout3')
            
            fullyConnectedLayer(k, 'Name', 'fc_final')
            sigmoidLayer('Name', 'sigmoid')
            regressionLayer('Name', 'output')];
        lgraph = addLayers(lgraph, deep_layers);
        lgraph = connectLayers(lgraph, 'combined_input', 'fc1');
        initial_lr = 0.001;
    end
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', initial_lr, ...
        'ValidationData', {X_val, Y_val}, ...
        'ValidationFrequency', 200, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch', ...
        'ValidationPatience', 6);
    
    % Train
    model = trainNetwork(X_train_final, Y_train_final, lgraph, options);
    
    % Save
    if ~exist('mat/universal', 'dir'), mkdir('mat/universal'); end
    save(sprintf('mat/universal/progressive_model_%s_tau%02d.mat', stage_name, tau_val*100), 'model');
    fprintf('Model saved for stage %s (τ=%.2f)\n', stage_name, tau_val);
end

% Include your helper functions here...
function [X, y_bits] = generate_universal_data(N_sym, tau, SNR_dB, mod_type, M, phase, win_len, num_past, num_future)
    k = log2(M);
    constellation = custom_modulator(M, mod_type, phase);
    bits = randi([0 1], N_sym * k, 1);
    symbols = map_bits_to_symbols(bits, k, constellation);
    sps=10; beta=0.3; span=6; h=rcosdesign(beta,span,sps,'sqrt');
    tx_up = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_up, h);
    pwr = mean(abs(txSignal).^2); txSignal = txSignal / sqrt(pwr);
    snr_lin = 10^(SNR_dB/10); noise_var = 1 / (2*snr_lin);
    noise = sqrt(noise_var/2) * (randn(size(txSignal)) + 1j*randn(size(txSignal)));
    rxSignal = txSignal + noise;
    rxMF = conv(rxSignal, h);
    delay = finddelay(tx_up, rxMF);
    
    X = zeros(N_sym, win_len*2 + (num_past+num_future)*2);
    y_bits = zeros(N_sym, k);
    half_win = floor(win_len/2);
    populated_rows = false(N_sym, 1);
    
    for i = (num_past + 1):(N_sym - num_future)
        loc = round((i-1)*sps*tau) + 1 + delay;
        if (loc > half_win) && (loc + half_win <= length(rxMF))
            win_c = rxMF(loc-half_win:loc+half_win);
            past_c = symbols(i-1:-1:i-num_past);
            future_c = symbols(i+1:i+num_future);
            
            win_ri = [real(win_c(:)'), imag(win_c(:)')];
            hist_ri = [real(past_c(:)'), imag(past_c(:)'), real(future_c(:)'), imag(future_c(:)')];
            X(i, :) = [win_ri, hist_ri];
            
            y_bits(i, :) = bits( (i-1)*k+1 : i*k )';
            populated_rows(i) = true;
        end
    end
    X = X(populated_rows,:);
    y_bits = y_bits(populated_rows,:);
end

function constellation = custom_modulator(M, type, phase)
    if strcmpi(type, 'psk'), p=0:M-1; constellation=exp(1j*(2*pi*p/M + phase));
    elseif strcmpi(type, 'qam'), k=log2(M); if mod(k,2)~=0, error('QAM M must be a square number.'); end; n=sqrt(M); vals=-(n-1):2:(n-1); [X,Y]=meshgrid(vals,vals); constellation=X+1j*Y; constellation = constellation * exp(1j*phase);
    else error('Unknown modulation type.'); end
    constellation = constellation(:);
    constellation = constellation / sqrt(mean(abs(constellation).^2));
end

function symbols = map_bits_to_symbols(bits, k, constellation)
    num_bits = length(bits);
    num_symbols = floor(num_bits/k);
    if num_symbols == 0, symbols = []; return; end
    bit_matrix = reshape(bits(1:num_symbols*k), k, num_symbols)';
    indices = bi2de(bit_matrix, 'left-msb') + 1;
    symbols = constellation(indices);
end