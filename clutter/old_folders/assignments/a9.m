clear, clc

% --- Parameters ---
N = 50000;
sps = 10;
tau = .8;
beta = .3;
span = 6;
window_len = 2*floor(3*sps/2)+1;
h = rcosdesign(beta, span, sps, 'sqrt');
pulse_loc = (length(h)-1)/2; % Correct way to calculate total delay of ONE filter

SNR_range = 0:2:10;
ber_fnn = zeros(size(SNR_range));
ber_cnn = ber_fnn;

% --- Define Architectures ---
fnn_layers = [
    featureInputLayer(window_len)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

cnn_layers = [
    sequenceInputLayer(1, 'Name', 'input') 
    
    convolution1dLayer(5, 32, 'Padding', 'causal') 
    reluLayer
    batchNormalizationLayer
    
    convolution1dLayer(7, 64, 'Padding', 'causal')
    reluLayer
    batchNormalizationLayer

    globalAveragePooling1dLayer
    
    fullyConnectedLayer(32)
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% --- Main Loop ---
for snridx = 1:length(SNR_range)
    current_SNR = SNR_range(snridx);
    fprintf('--- Running SNR = %d dB ---\n', current_SNR);
    
    % --- Data Generation ---
    bits = randi([0 1], N, 1);
    symbols = 1-2*bits;
    tx_upsampled = upsample(symbols, round(sps*tau));
    txSignal = conv(tx_upsampled, h);
    rxSignal = awgn(txSignal, current_SNR, 'measured');
    rxMF = conv(rxSignal, h);

    xdata = zeros(N, window_len);
    start_idx = 2*pulse_loc + 1;
    total_delay_win = floor(window_len/2);
    ploc = start_idx;
    
    for i = 1 : N
        if (ploc - total_delay_win > 0) && (ploc + total_delay_win <= length(rxMF))
            xdata(i, :) = rxMF(ploc - total_delay_win : ploc + total_delay_win);
        end
        ploc = ploc + sps*tau;
    end
    ydata = bits;
    Y_categorical = categorical(ydata);
    
    % --- Data Preparation ---
    cv = cvpartition(N, 'HoldOut', 0.2);
    
    % Data for FNN (standard matrix)
    X_train_fnn = xdata(cv.training, :);
    Y_train = Y_categorical(cv.training, :); % Both networks use the same label format now
    X_val_fnn = xdata(cv.test, :);
    Y_val = Y_categorical(cv.test, :);
    
    % Data for CNN (cell array for predictors, standard array for labels)
    X_train_cnn = num2cell(X_train_fnn, 2);
    X_val_cnn = num2cell(X_val_fnn, 2);

    % --- Training Options ---
    options_fnn = trainingOptions('adam', 'MaxEpochs', 10, 'MiniBatchSize', 128, 'ValidationData', {X_val_fnn, Y_val}, 'GradientThreshold', 1, 'Verbose', false, 'Plots', 'none');
    options_cnn = trainingOptions('adam', 'MaxEpochs', 10, 'MiniBatchSize', 128, 'ValidationData', {X_val_cnn, Y_val}, 'GradientThreshold', 1, 'Verbose', false, 'Plots', 'none');

    % --- Train and Evaluate FNN ---
    fprintf('Training FNN...\n');
    ftn_detector_fnn = trainNetwork(X_train_fnn, Y_train, fnn_layers, options_fnn);
    Y_pred_fnn = classify(ftn_detector_fnn, X_val_fnn);
    ber_fnn(snridx) = 1 - sum(Y_pred_fnn == Y_val) / numel(Y_val);

    % --- Train and Evaluate CNN ---
    fprintf('Training CNN...\n');
    ftn_detector_cnn = trainNetwork(X_train_cnn, Y_train, cnn_layers, options_cnn);
    Y_pred_cnn = classify(ftn_detector_cnn, X_val_cnn);
    ber_cnn(snridx) = 1 - sum(Y_pred_cnn == Y_val) / numel(Y_val);
    
    fprintf('FNN BER: %.5f | CNN BER: %.5f\n', ber_fnn(snridx), ber_cnn(snridx));
end

% --- Final Plot ---
figure;
semilogy(SNR_range, ber_fnn, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8)
hold on, grid on
semilogy(SNR_range, ber_cnn, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8)
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('FNN Detector', 'CNN Detector');
title('FTN Detector Performance Comparison');
ylim([1e-4 0.5]);