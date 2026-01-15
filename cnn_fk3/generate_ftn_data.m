function [X_train, Y_train, X_test, Y_test] = generate_ftn_data(tau, N_train, N_test, SNR_values, beta)
%GENERATE_FTN_DATA Generate training and test data for CNN-FK3
%   Based on the paper's methodology (Section III-A, Table IV)
%
%   Inputs:
%       tau        - FTN compression factor
%       N_train    - Number of training samples
%       N_test     - Number of test samples  
%       SNR_values - Array of SNR values in dB (e.g., [7, 8, 9, 10])
%       beta       - Roll-off factor (default: 0.35)
%
%   Outputs:
%       X_train, Y_train - Training data (features and labels)
%       X_test, Y_test   - Test data (features and labels)

if nargin < 5
    beta = 0.35;
end

%% Determine N based on tau (Table III)
switch tau
    case 0.9
        N = 2;
    case 0.8
        N = 6;
    case 0.7
        N = 8;
    otherwise
        error('Unsupported tau value');
end

input_size = 2*N + 1;
fprintf('Generating FTN data for tau=%.1f, N=%d\n', tau, N);

%% SRRC pulse parameters
sps = 10;               % Samples per symbol (Nyquist)
span = 6;               % Filter span
step = round(tau * sps); % Samples per FTN symbol

% Generate SRRC pulse
h = rcosdesign(beta, span, sps, 'sqrt');
h = h / norm(h);

% Matched filter output (for ISI coefficients reference)
hh = conv(h, h);
[~, pk] = max(hh);
delay = span * sps;

%% Preallocate
samples_per_snr = ceil(N_train / length(SNR_values));
X_train = zeros(input_size, N_train);
Y_train = zeros(1, N_train);

sample_idx = 1;

fprintf('Generating training data (%d samples)...\n', N_train);

for snr_idx = 1:length(SNR_values)
    SNR_dB = SNR_values(snr_idx);
    EbN0_lin = 10^(SNR_dB / 10);
    N0 = 1 / EbN0_lin;
    noise_std = sqrt(N0 / 2);
    
    % Generate data in batches
    batch_size = 10000;  % Symbols per batch
    num_batches = ceil(samples_per_snr / batch_size);
    
    for batch = 1:num_batches
        if sample_idx > N_train
            break;
        end
        
        % Generate random bits
        bits = randi([0 1], batch_size, 1);
        symbols = 2*bits - 1;  % BPSK: 0->-1, 1->+1
        
        % Pad with -1
        symbols_padded = [-ones(N, 1); symbols; -ones(N, 1)];
        
        % Upsample and filter
        tx_up = zeros(length(symbols_padded) * step, 1);
        tx_up(1:step:end) = symbols_padded;
        tx_sig = conv(tx_up, h);
        
        % Add noise
        noise = noise_std * randn(size(tx_sig));
        rx_sig = tx_sig + noise;
        
        % Matched filter
        rx_mf = conv(rx_sig, h);
        
        % Sample at symbol rate
        total_delay = 2 * delay;
        
        % Extract windows for each symbol
        for k = 1:batch_size
            if sample_idx > N_train
                break;
            end
            
            % Center sample index
            center_idx = total_delay + (N + k - 1) * step + 1;
            
            % Extract window of 2N+1 samples at symbol rate
            window = zeros(input_size, 1);
            for w = 1:input_size
                sym_offset = w - (N + 1);  % -N to +N
                sample_pos = center_idx + sym_offset * step;
                if sample_pos > 0 && sample_pos <= length(rx_mf)
                    window(w) = rx_mf(sample_pos);
                end
            end
            
            X_train(:, sample_idx) = window;
            Y_train(sample_idx) = bits(k);  % Label is 0 or 1
            sample_idx = sample_idx + 1;
        end
    end
    
    fprintf('  SNR=%d dB: %d samples generated\n', SNR_dB, min(sample_idx-1, N_train));
end

% Trim if overgenerated
X_train = X_train(:, 1:min(sample_idx-1, N_train));
Y_train = Y_train(1:min(sample_idx-1, N_train));

%% Generate test data (similar process)
fprintf('Generating test data (%d samples)...\n', N_test);

samples_per_snr_test = ceil(N_test / length(SNR_values));
X_test = zeros(input_size, N_test);
Y_test = zeros(1, N_test);

sample_idx = 1;

for snr_idx = 1:length(SNR_values)
    SNR_dB = SNR_values(snr_idx);
    EbN0_lin = 10^(SNR_dB / 10);
    N0 = 1 / EbN0_lin;
    noise_std = sqrt(N0 / 2);
    
    batch_size = 10000;
    num_batches = ceil(samples_per_snr_test / batch_size);
    
    for batch = 1:num_batches
        if sample_idx > N_test
            break;
        end
        
        bits = randi([0 1], batch_size, 1);
        symbols = 2*bits - 1;
        symbols_padded = [-ones(N, 1); symbols; -ones(N, 1)];
        
        tx_up = zeros(length(symbols_padded) * step, 1);
        tx_up(1:step:end) = symbols_padded;
        tx_sig = conv(tx_up, h);
        
        noise = noise_std * randn(size(tx_sig));
        rx_sig = tx_sig + noise;
        rx_mf = conv(rx_sig, h);
        
        total_delay = 2 * delay;
        
        for k = 1:batch_size
            if sample_idx > N_test
                break;
            end
            
            center_idx = total_delay + (N + k - 1) * step + 1;
            
            window = zeros(input_size, 1);
            for w = 1:input_size
                sym_offset = w - (N + 1);
                sample_pos = center_idx + sym_offset * step;
                if sample_pos > 0 && sample_pos <= length(rx_mf)
                    window(w) = rx_mf(sample_pos);
                end
            end
            
            X_test(:, sample_idx) = window;
            Y_test(sample_idx) = bits(k);
            sample_idx = sample_idx + 1;
        end
    end
end

X_test = X_test(:, 1:min(sample_idx-1, N_test));
Y_test = Y_test(1:min(sample_idx-1, N_test));

fprintf('Data generation complete.\n');
fprintf('  Training: %d samples\n', size(X_train, 2));
fprintf('  Test: %d samples\n', size(X_test, 2));

end
