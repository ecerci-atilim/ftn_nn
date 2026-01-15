%% CNN-FK3 Training Script (Fixed Gradient Version)
%  Exact implementation of Table IV hyperparameters
%  Tokluoglu et al., IEEE Trans. Commun., 2025
%
%  Author: Emre Cerci
%  Date: January 2026

clear; clc; close all;

%% ========== CONFIGURATION (Table IV) ==========
tau = 0.7;                      % FTN compression factor

% Training hyperparameters
N_train = 4000000;              % Number of training samples
N_test = 4000000;               % Number of test samples
SNR_values = [7, 8, 9, 10];     % SNR values during training (dB)
batch_size = 1000;              % Batch size
num_epochs = 20;                % Number of epochs
steps_per_epoch = 4000;         % Steps per epoch

% Adam optimizer parameters
initial_lr = 0.001;             % Initial learning rate
beta1 = 0.9;                    % Adam beta1
beta2 = 0.999;                  % Adam beta2
epsilon = 1e-8;                 % Adam epsilon

% Learning rate schedule (Exponential Decay)
decay_interval_epochs = 5;      % Decay every 5 epochs
decay_rate = 0.9;               % Decay rate

% Roll-off factor
beta = 0.35;

%% ========== DETERMINE N ==========
switch tau
    case 0.9
        N = 2;
        filters_per_layer = [2, 1];
    case 0.8
        N = 6;
        filters_per_layer = [4, 2, 2, 1, 1, 1];
    case 0.7
        N = 8;
        filters_per_layer = [8, 6, 4, 2, 2, 1, 1, 1];
end

num_fk_layers = N;
total_filters = sum(filters_per_layer);
input_size = 2*N + 1;

fprintf('============================================\n');
fprintf('CNN-FK3 Training for tau=%.1f\n', tau);
fprintf('============================================\n');
fprintf('N=%d, Input size=%d, Total filters=%d\n\n', N, input_size, total_filters);

%% ========== DATA GENERATION ==========
fprintf('Generating training data...\n');
tic;

% For faster initial testing, reduce dataset size
N_train_actual = min(N_train, 1000000);
N_test_actual = min(N_test, 200000);
% N_train_actual = N_train;
% N_test_actual = N_train;

[X_train, Y_train, X_test, Y_test] = generate_ftn_data(tau, N_train_actual, N_test_actual, SNR_values, beta);

data_gen_time = toc;
fprintf('Data generation completed in %.1f seconds\n\n', data_gen_time);

%% ========== INITIALIZE PARAMETERS ==========
fprintf('Initializing model parameters...\n');

% Store all parameters in a flat struct for dlgradient compatibility
params = struct();

% FK layer weights and biases (flattened naming)
for i = 1:num_fk_layers
    F_i = filters_per_layer(i);
    scale = sqrt(2 / (3 + F_i));
    params.(sprintf('W_fk%d', i)) = dlarray(scale * randn(3, F_i, 'single'));
    params.(sprintf('b_fk%d', i)) = dlarray(zeros(F_i, 1, 'single'));
end

% Dense layer
scale_dense = sqrt(2 / (total_filters + 4));
params.W_dense = dlarray(scale_dense * randn(total_filters, 4, 'single'));
params.b_dense = dlarray(zeros(4, 1, 'single'));

% Output layer
scale_out = sqrt(2 / 5);
params.W_out = dlarray(scale_out * randn(4, 1, 'single'));
params.b_out = dlarray(zeros(1, 1, 'single'));

% Count parameters
total_params = 0;
param_names = fieldnames(params);
for i = 1:length(param_names)
    total_params = total_params + numel(params.(param_names{i}));
end
fprintf('Total parameters: %d\n\n', total_params);

%% ========== INITIALIZE ADAM STATE ==========
adam_m = struct();
adam_v = struct();

for i = 1:length(param_names)
    name = param_names{i};
    adam_m.(name) = dlarray(zeros(size(params.(name)), 'single'));
    adam_v.(name) = dlarray(zeros(size(params.(name)), 'single'));
end

%% ========== TRAINING LOOP ==========
fprintf('Starting training...\n');
fprintf('Epochs: %d, Batch size: %d, Initial LR: %.4f\n\n', num_epochs, batch_size, initial_lr);

num_samples = size(X_train, 2);
num_batches = floor(num_samples / batch_size);

% Training history
history.loss = [];
history.val_loss = [];
history.lr = [];

global_step = 0;
current_lr = initial_lr;

for epoch = 1:num_epochs
    epoch_start = tic;
    
    % Learning rate decay
    if epoch > 1 && mod(epoch-1, decay_interval_epochs) == 0
        current_lr = current_lr * decay_rate;
        fprintf('  Learning rate decayed to %.6f\n', current_lr);
    end
    
    % Shuffle data
    perm = randperm(num_samples);
    X_shuffled = X_train(:, perm);
    Y_shuffled = Y_train(perm);
    
    epoch_loss = 0;
    actual_batches = min(num_batches, steps_per_epoch);
    
    for batch_idx = 1:actual_batches
        global_step = global_step + 1;
        
        % Get batch
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = start_idx + batch_size - 1;
        
        X_batch = dlarray(single(X_shuffled(:, start_idx:end_idx)));
        Y_batch = dlarray(single(Y_shuffled(start_idx:end_idx)));
        
        % Compute gradients
        [loss, grads] = dlfeval(@computeGradients, params, X_batch, Y_batch, ...
                                N, num_fk_layers, filters_per_layer);
        
        % Adam update
        for i = 1:length(param_names)
            name = param_names{i};
            
            % Update momentum
            adam_m.(name) = beta1 * adam_m.(name) + (1 - beta1) * grads.(name);
            adam_v.(name) = beta2 * adam_v.(name) + (1 - beta2) * (grads.(name).^2);
            
            % Bias correction
            m_hat = adam_m.(name) / (1 - beta1^global_step);
            v_hat = adam_v.(name) / (1 - beta2^global_step);
            
            % Update parameter
            params.(name) = params.(name) - current_lr * m_hat ./ (sqrt(v_hat) + epsilon);
        end
        
        epoch_loss = epoch_loss + double(extractdata(loss));
        
        % Progress
        if mod(batch_idx, 500) == 0
            fprintf('  Epoch %d, Batch %d/%d, Loss: %.6f\n', ...
                    epoch, batch_idx, actual_batches, epoch_loss / batch_idx);
        end
    end
    
    avg_loss = epoch_loss / actual_batches;
    
    % Validation
    val_size = min(10000, size(X_test, 2));
    X_val = dlarray(single(X_test(:, 1:val_size)));
    Y_val = dlarray(single(Y_test(1:val_size)));
    val_loss = double(extractdata(computeLoss(params, X_val, Y_val, N, num_fk_layers, filters_per_layer)));
    
    epoch_time = toc(epoch_start);
    
    fprintf('Epoch %d/%d - Loss: %.6f, Val Loss: %.6f, LR: %.6f, Time: %.1fs\n', ...
            epoch, num_epochs, avg_loss, val_loss, current_lr, epoch_time);
    
    history.loss(epoch) = avg_loss;
    history.val_loss(epoch) = val_loss;
    history.lr(epoch) = current_lr;
end

%% ========== SAVE MODEL ==========
config.tau = tau;
config.N = N;
config.filters_per_layer = filters_per_layer;
config.num_fk_layers = num_fk_layers;
config.total_filters = total_filters;

model_filename = sprintf('cnn_fk3_tau%.1f_%s.mat', tau, datestr(now, 'yyyymmdd_HHMM'));
save(model_filename, 'params', 'config', 'history');
fprintf('\nModel saved to %s\n', model_filename);

%% ========== PLOT ==========
figure('Position', [100 100 800 400]);

subplot(1,2,1);
plot(1:num_epochs, history.loss, 'b-', 'LineWidth', 2);
hold on;
plot(1:num_epochs, history.val_loss, 'r--', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title(sprintf('Training Loss (\\tau=%.1f)', tau));
legend('Train', 'Val'); grid on;

subplot(1,2,2);
plot(1:num_epochs, history.lr, 'g-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Learning Rate');
title('LR Schedule'); grid on;

saveas(gcf, sprintf('training_curves_tau%.1f.png', tau));

fprintf('\nTraining completed.\n');

%% ========== FUNCTIONS ==========

function [loss, grads] = computeGradients(params, X, Y, N, num_fk_layers, filters_per_layer)
    % Forward pass
    y_pred = forwardPass(params, X, N, num_fk_layers, filters_per_layer);
    
    % Loss
    eps_val = 1e-7;
    y_pred = max(min(y_pred, 1-eps_val), eps_val);
    loss = -mean(Y .* log(y_pred) + (1 - Y) .* log(1 - y_pred), 'all');
    
    % Gradients
    grads = struct();
    
    % FK layers
    for i = 1:num_fk_layers
        grads.(sprintf('W_fk%d', i)) = dlgradient(loss, params.(sprintf('W_fk%d', i)));
        grads.(sprintf('b_fk%d', i)) = dlgradient(loss, params.(sprintf('b_fk%d', i)));
    end
    
    % Dense and output
    grads.W_dense = dlgradient(loss, params.W_dense);
    grads.b_dense = dlgradient(loss, params.b_dense);
    grads.W_out = dlgradient(loss, params.W_out);
    grads.b_out = dlgradient(loss, params.b_out);
end

function loss = computeLoss(params, X, Y, N, num_fk_layers, filters_per_layer)
    y_pred = forwardPass(params, X, N, num_fk_layers, filters_per_layer);
    eps_val = 1e-7;
    y_pred = max(min(y_pred, 1-eps_val), eps_val);
    loss = -mean(Y .* log(y_pred) + (1 - Y) .* log(1 - y_pred), 'all');
end

function y = forwardPass(params, X, N, num_fk_layers, filters_per_layer)
    center = N + 1;
    
    % FK layers
    fk_outputs = cell(num_fk_layers, 1);
    for i = 1:num_fk_layers
        idx_left = center - i;
        idx_right = center + i;
        triplet = X([idx_left, center, idx_right], :);  % [3 x batch]
        
        W = params.(sprintf('W_fk%d', i));  % [3 x F_i]
        b = params.(sprintf('b_fk%d', i));  % [F_i x 1]
        
        z = W' * triplet + b;  % [F_i x batch]
        fk_outputs{i} = tanh(z);
    end
    
    % Concatenate
    concat_out = cat(1, fk_outputs{:});  % [total_filters x batch]
    
    % Dense
    z_dense = params.W_dense' * concat_out + params.b_dense;
    h_dense = tanh(z_dense);
    
    % Output
    z_out = params.W_out' * h_dense + params.b_out;
    y = sigmoid(z_out);
end