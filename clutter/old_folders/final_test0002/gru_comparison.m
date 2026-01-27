%% GRU Comparison: Symbol-Rate vs Window (Fair Span)
clear; clc; close all;

%% Parameters
tau = 0.7;
SNR_dB = 10;
N_train = 50000;
N_test = 20000;
sps = 10;
beta = 0.3;
span = 6;

step = round(tau * sps);
EbN0 = 10^(SNR_dB/10);
noise_var = 1 / (2 * EbN0);

L_sym = 4;
win_radius = L_sym * step;

fprintf('tau=%.1f, step=%d, window radius=%d\n', tau, step, win_radius);
fprintf('Method 1: %d points (symbol-rate)\n', 2*L_sym+1);
fprintf('Method 2: %d points (fractional)\n', 2*win_radius+1);

%% Generate Pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);
delay = span * sps;

%% Training Data
rng(42);
bits_train = randi([0 1], N_train, 1);
symbols_train = 2*bits_train - 1;

tx = conv(upsample(symbols_train, step), h);
rx_noisy = tx + sqrt(noise_var) * randn(size(tx));
rx_mf = conv(rx_noisy, h);
rx_mf = rx_mf / std(rx_mf);

%% Test Data
rng(123);
bits_test = randi([0 1], N_test, 1);
symbols_test = 2*bits_test - 1;

tx_test = conv(upsample(symbols_test, step), h);
rx_noisy_test = tx_test + sqrt(noise_var) * randn(size(tx_test));
rx_mf_test = conv(rx_noisy_test, h);
rx_mf_test = rx_mf_test / std(rx_mf_test);

%% Method 1: Symbol-Rate (9 points, span=57)
offsets1 = (-L_sym:L_sym) * step;
X1_train_cell = cell(N_train, 1);
X1_test_cell = cell(N_test, 1);

for k = 1:N_train
    center = (k-1)*step + 1 + delay;
    seq = rx_mf(center + offsets1);
    X1_train_cell{k} = reshape(seq, 1, []);  % (1 x 9)
end
for k = 1:N_test
    center = (k-1)*step + 1 + delay;
    seq = rx_mf_test(center + offsets1);
    X1_test_cell{k} = reshape(seq, 1, []);
end

%% Method 2: Window (57 points, span=57)
offsets2 = -win_radius:win_radius;
X2_train_cell = cell(N_train, 1);
X2_test_cell = cell(N_test, 1);

for k = 1:N_train
    center = (k-1)*step + 1 + delay;
    seq = rx_mf(center + offsets2);
    X2_train_cell{k} = reshape(seq, 1, []);  % (1 x 57)
end
for k = 1:N_test
    center = (k-1)*step + 1 + delay;
    seq = rx_mf_test(center + offsets2);
    X2_test_cell{k} = reshape(seq, 1, []);
end

%% Labels
Y_train = categorical(bits_train);
Y_test = categorical(bits_test);

%% GRU Models
layers1 = [
    sequenceInputLayer(1)
    gruLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

layers2 = [
    sequenceInputLayer(1)
    gruLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options1 = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'none');

options2 = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'none');

%% Train
fprintf('\n=== Method 1: Symbol-Rate GRU (9 timesteps) ===\n');
net1 = trainNetwork(X1_train_cell, Y_train, layers1, options1);

fprintf('\n=== Method 2: Window GRU (%d timesteps) ===\n', 2*win_radius+1);
net2 = trainNetwork(X2_train_cell, Y_train, layers2, options2);

%% Evaluate
pred1 = classify(net1, X1_test_cell);
pred2 = classify(net2, X2_test_cell);

BER1 = mean(pred1 ~= Y_test);
BER2 = mean(pred2 ~= Y_test);
ACC1 = 100 * (1 - BER1);
ACC2 = 100 * (1 - BER2);

%% Results
fprintf('\n========== RESULTS (tau=%.1f, SNR=%d dB) ==========\n', tau, SNR_dB);
fprintf('Method 1 - Symbol-Rate (%d pts, span=%d)\n', 2*L_sym+1, 2*win_radius+1);
fprintf('   BER = %.4e,  Accuracy = %.2f%%\n', BER1, ACC1);
fprintf('Method 2 - Window (%d pts, span=%d)\n', 2*win_radius+1, 2*win_radius+1);
fprintf('   BER = %.4e,  Accuracy = %.2f%%\n', BER2, ACC2);
fprintf('===================================================\n');

save('gru_comparison_results.mat', 'BER1', 'BER2', 'ACC1', 'ACC2', 'tau', 'SNR_dB');