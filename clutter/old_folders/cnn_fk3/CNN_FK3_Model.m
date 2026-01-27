classdef CNN_FK3_Model < handle
    %CNN_FK3_MODEL Structured Fixed Kernel CNN for FTN Detection
    %   Exact implementation of Tokluoglu et al., IEEE TCOM 2025
    %
    %   Architecture:
    %   - N Fixed Kernel Layers (each processes triplet [y_{k-i}, y_k, y_{k+i}])
    %   - Concatenation of all filter outputs
    %   - Dense layer (4 neurons, tanh)
    %   - Output layer (1 neuron, sigmoid)
    
    properties
        tau             % FTN compression factor
        N               % One-sided ISI length
        input_size      % 2N+1
        filters_per_layer  % Filter count for each FK layer
        num_fk_layers   % Number of fixed kernel layers
        total_filters   % Sum of all filters
        
        % Learnable parameters (stored as dlarray for auto-diff)
        W_fk            % Cell array of FK layer weights {3 x F_i}
        b_fk            % Cell array of FK layer biases {F_i x 1}
        W_dense         % Dense layer weights {total_filters x 4}
        b_dense         % Dense layer bias {4 x 1}
        W_out           % Output layer weights {4 x 1}
        b_out           % Output layer bias {1 x 1}
    end
    
    methods
        function obj = CNN_FK3_Model(tau)
            %CNN_FK3_MODEL Constructor
            obj.tau = tau;
            
            % Set N and filters based on tau (Table III)
            switch tau
                case 0.9
                    obj.N = 2;
                    obj.filters_per_layer = [2, 1];
                case 0.8
                    obj.N = 6;
                    obj.filters_per_layer = [4, 2, 2, 1, 1, 1];
                case 0.7
                    obj.N = 8;
                    obj.filters_per_layer = [8, 6, 4, 2, 2, 1, 1, 1];
                otherwise
                    error('Unsupported tau. Use 0.7, 0.8, or 0.9');
            end
            
            obj.input_size = 2*obj.N + 1;
            obj.num_fk_layers = obj.N;
            obj.total_filters = sum(obj.filters_per_layer);
            
            % Initialize weights (Xavier/Glorot initialization)
            obj.initializeWeights();
            
            fprintf('CNN-FK3 Model created for tau=%.1f\n', tau);
            fprintf('  N=%d, Input size=%d, Total filters=%d\n', ...
                    obj.N, obj.input_size, obj.total_filters);
            fprintf('  Total parameters: %d\n', obj.countParameters());
        end
        
        function initializeWeights(obj)
            %INITIALIZEWEIGHTS Xavier/Glorot initialization
            
            obj.W_fk = cell(obj.num_fk_layers, 1);
            obj.b_fk = cell(obj.num_fk_layers, 1);
            
            for i = 1:obj.num_fk_layers
                F_i = obj.filters_per_layer(i);
                % Xavier init for 3 inputs -> F_i outputs
                scale = sqrt(2 / (3 + F_i));
                obj.W_fk{i} = dlarray(scale * randn(3, F_i, 'single'));
                obj.b_fk{i} = dlarray(zeros(F_i, 1, 'single'));
            end
            
            % Dense layer: total_filters -> 4
            scale_dense = sqrt(2 / (obj.total_filters + 4));
            obj.W_dense = dlarray(scale_dense * randn(obj.total_filters, 4, 'single'));
            obj.b_dense = dlarray(zeros(4, 1, 'single'));
            
            % Output layer: 4 -> 1
            scale_out = sqrt(2 / (4 + 1));
            obj.W_out = dlarray(scale_out * randn(4, 1, 'single'));
            obj.b_out = dlarray(zeros(1, 1, 'single'));
        end
        
        function y = forward(obj, x)
            %FORWARD Forward pass through the network
            %   x: input dlarray of size [input_size x batch_size]
            %   y: output dlarray of size [1 x batch_size]
            
            batch_size = size(x, 2);
            center = obj.N + 1;  % Index of center symbol (1-indexed)
            
            % Process each Fixed Kernel layer
            fk_outputs = cell(obj.num_fk_layers, 1);
            
            for i = 1:obj.num_fk_layers
                % Extract triplet [y_{k-i}, y_k, y_{k+i}]
                idx_left = center - i;
                idx_right = center + i;
                
                triplet = x([idx_left, center, idx_right], :);  % [3 x batch]
                
                % Linear transformation: triplet' * W + b
                % triplet: [3 x batch], W: [3 x F_i]
                % Result: [batch x F_i] -> transpose to [F_i x batch]
                z = obj.W_fk{i}' * triplet + obj.b_fk{i};  % [F_i x batch]
                
                % Tanh activation (Eq. 15 in paper)
                fk_outputs{i} = tanh(z);
            end
            
            % Concatenate all FK outputs
            concat_out = cat(1, fk_outputs{:});  % [total_filters x batch]
            
            % Dense layer (Eq. 16)
            z_dense = obj.W_dense' * concat_out + obj.b_dense;  % [4 x batch]
            h_dense = tanh(z_dense);
            
            % Output layer (Eq. 20)
            z_out = obj.W_out' * h_dense + obj.b_out;  % [1 x batch]
            y = sigmoid(z_out);
        end
        
        function loss = computeLoss(obj, x, y_true)
            %COMPUTELOSS Binary cross-entropy loss
            %   x: input [input_size x batch]
            %   y_true: labels [1 x batch] (0 or 1)
            
            y_pred = obj.forward(x);
            
            % Binary cross-entropy with numerical stability
            eps_val = 1e-7;
            y_pred = max(min(y_pred, 1-eps_val), eps_val);
            
            loss = -mean(y_true .* log(y_pred) + (1 - y_true) .* log(1 - y_pred), 'all');
        end
        
        function [loss, gradients] = computeGradients(obj, x, y_true)
            %COMPUTEGRADIENTS Compute gradients using automatic differentiation
            
            [loss, gradients] = dlfeval(@obj.lossWithGrad, x, y_true);
        end
        
        function [loss, grads] = lossWithGrad(obj, x, y_true)
            %LOSSWITHGRAD Helper for gradient computation
            
            y_pred = obj.forward(x);
            
            eps_val = 1e-7;
            y_pred = max(min(y_pred, 1-eps_val), eps_val);
            
            loss = -mean(y_true .* log(y_pred) + (1 - y_true) .* log(1 - y_pred), 'all');
            
            % Compute gradients w.r.t. all parameters
            grads = struct();
            
            % This is a placeholder - actual gradients computed via dlgradient
            % in the training loop using dlfeval
        end
        
        function y_pred = predict(obj, x)
            %PREDICT Predict without gradient tracking
            
            if ~isa(x, 'dlarray')
                x = dlarray(single(x));
            end
            
            y = obj.forward(x);
            y_pred = double(extractdata(y));
        end
        
        function bits = decode(obj, x, threshold)
            %DECODE Hard decision decoding
            
            if nargin < 3
                threshold = 0.5;
            end
            
            y_pred = obj.predict(x);
            bits = double(y_pred > threshold);
        end
        
        function n = countParameters(obj)
            %COUNTPARAMETERS Count total learnable parameters
            
            n = 0;
            for i = 1:obj.num_fk_layers
                n = n + numel(obj.W_fk{i}) + numel(obj.b_fk{i});
            end
            n = n + numel(obj.W_dense) + numel(obj.b_dense);
            n = n + numel(obj.W_out) + numel(obj.b_out);
        end
        
        function params = getParameters(obj)
            %GETPARAMETERS Get all parameters as a struct
            
            params.W_fk = obj.W_fk;
            params.b_fk = obj.b_fk;
            params.W_dense = obj.W_dense;
            params.b_dense = obj.b_dense;
            params.W_out = obj.W_out;
            params.b_out = obj.b_out;
        end
        
        function setParameters(obj, params)
            %SETPARAMETERS Set all parameters from a struct
            
            obj.W_fk = params.W_fk;
            obj.b_fk = params.b_fk;
            obj.W_dense = params.W_dense;
            obj.b_dense = params.b_dense;
            obj.W_out = params.W_out;
            obj.b_out = params.b_out;
        end
        
        function saveModel(obj, filename)
            %SAVEMODEL Save model to file
            
            params = obj.getParameters();
            config.tau = obj.tau;
            config.N = obj.N;
            config.filters_per_layer = obj.filters_per_layer;
            
            save(filename, 'params', 'config');
            fprintf('Model saved to %s\n', filename);
        end
        
        function loadModel(obj, filename)
            %LOADMODEL Load model from file
            
            data = load(filename);
            obj.setParameters(data.params);
            fprintf('Model loaded from %s\n', filename);
        end
    end
end
