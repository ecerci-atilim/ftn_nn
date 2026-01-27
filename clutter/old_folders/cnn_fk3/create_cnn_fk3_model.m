function net = create_cnn_fk3_model(tau)
%CREATE_CNN_FK3_MODEL Create CNN-FK3 model for FTN detection
%   Exact implementation of:
%   "A Novel CNN-Based Standalone Detector for Faster-Than-Nyquist Signaling"
%   Tokluoglu et al., IEEE Trans. Commun., 2025
%
%   Input:
%       tau - FTN compression factor (0.7, 0.8, or 0.9)
%
%   Output:
%       net - dlnetwork object

%% Determine N and filter allocation based on tau (Table III)
switch tau
    case 0.9
        N = 2;
        filters_per_layer = [2, 1];  % Layer 1: 2 filters, Layer 2: 1 filter
    case 0.8
        N = 6;
        filters_per_layer = [4, 2, 2, 1, 1, 1];  % 6 layers
    case 0.7
        N = 8;
        filters_per_layer = [8, 6, 4, 2, 2, 1, 1, 1];  % 8 layers
    otherwise
        error('Unsupported tau value. Use 0.7, 0.8, or 0.9');
end

input_size = 2*N + 1;
num_layers = N;
total_filters = sum(filters_per_layer);

fprintf('Creating CNN-FK3 model for tau=%.1f\n', tau);
fprintf('  N = %d (one-sided ISI length)\n', N);
fprintf('  Input size = %d\n', input_size);
fprintf('  Number of fixed kernel layers = %d\n', num_layers);
fprintf('  Filters per layer: [%s]\n', num2str(filters_per_layer));
fprintf('  Total features = %d\n', total_filters);

%% Build network using layer graph
layers = [
    % Input layer
    featureInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
];

% Create Fixed Kernel Layers using custom extraction + dense
% Each layer extracts [y_{k-i}, y_k, y_{k+i}] and applies weights

concat_names = cell(1, num_layers);

for i = 1:num_layers
    layer_name = sprintf('fk_layer_%d', i);
    
    % Indices for triplet: [y_{k-i}, y_k, y_{k+i}]
    % In 1-indexed MATLAB with center at N+1:
    % y_{k-i} -> index (N+1) - i = N+1-i
    % y_k     -> index N+1
    % y_{k+i} -> index (N+1) + i = N+1+i
    
    idx_left = N + 1 - i;
    idx_center = N + 1;
    idx_right = N + 1 + i;
    
    % Custom function layer to extract triplet
    extract_name = sprintf('extract_%d', i);
    
    % Fully connected layer for this fixed kernel (3 inputs -> F_i outputs)
    fc_name = sprintf('fc_fk_%d', i);
    act_name = sprintf('tanh_fk_%d', i);
    
    % Add extraction layer (custom)
    layers = [layers
        functionLayer(@(x) extractTriplet(x, idx_left, idx_center, idx_right), ...
            'Name', extract_name, ...
            'Formattable', true, ...
            'Acceleratable', true)
        fullyConnectedLayer(filters_per_layer(i), 'Name', fc_name)
        tanhLayer('Name', act_name)
    ];
    
    concat_names{i} = act_name;
end

%% Create layer graph and add concatenation
lgraph = layerGraph(layers(1));  % Start with input

% Add each branch separately
for i = 1:num_layers
    idx_start = 2 + (i-1)*3;  % Each branch has 3 layers
    branch_layers = layers(idx_start:idx_start+2);
    lgraph = addLayers(lgraph, branch_layers);
    lgraph = connectLayers(lgraph, 'input', sprintf('extract_%d', i));
end

% Add concatenation layer
lgraph = addLayers(lgraph, concatenationLayer(1, 'Name', 'concat'));

% Connect all branches to concatenation
for i = 1:num_layers
    lgraph = connectLayers(lgraph, concat_names{i}, sprintf('concat/in%d', i));
end

% Add Dense layer (4 neurons with tanh) and Output layer
lgraph = addLayers(lgraph, [
    fullyConnectedLayer(4, 'Name', 'dense')
    tanhLayer('Name', 'tanh_dense')
    fullyConnectedLayer(1, 'Name', 'output_fc')
    sigmoidLayer('Name', 'sigmoid')
]);

lgraph = connectLayers(lgraph, 'concat', 'dense');

%% Convert to dlnetwork
net = dlnetwork(lgraph);

fprintf('Model created successfully.\n');
fprintf('Total learnable parameters: %d\n', countParams(net));

end

%% Helper function to extract triplet
function y = extractTriplet(x, idx_left, idx_center, idx_right)
    % x is CBT format (Channel x Batch x Time) or just features
    % Extract indices [idx_left, idx_center, idx_right]
    y = x([idx_left, idx_center, idx_right], :);
end

%% Count parameters
function n = countParams(net)
    n = 0;
    for i = 1:numel(net.Learnables.Value)
        n = n + numel(net.Learnables.Value{i});
    end
end
