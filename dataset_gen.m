clear, clc

K = 3;
N = 15;
dataset_size = 80000;
tau = 0.5;
SNR_dB = 5;

window_len = 2*N + 1;

sps = 10;
span = 6;
rolloff = 0.3;
h = rcosdesign(rolloff, span, sps, 'sqrt');
h = h / norm(h);

bits = randi([0 1], dataset_size, 1);
symbols = 2*bits - 1;

tx = upsample(symbols, round(sps*tau));
tx = conv(tx, h);

EbN0_linear = 10^(SNR_dB/10);
noise_var = 1 / (2 * EbN0_linear);
noise = sqrt(noise_var) * randn(size(tx));
rx = tx + noise;

rx = conv(rx, h);

delay_system = span * sps;
symbol_spacing_samples = round(sps*tau);

half_win = N;

total_samples = K + window_len + K;
X = zeros(dataset_size, total_samples);
Y = zeros(dataset_size, 1);

valid_count = 0;
for i = 1:dataset_size
    current_symbol_idx = (i-1)*symbol_spacing_samples + 1 + delay_system;
    
    past_symbol_indices = current_symbol_idx - (1:K) * symbol_spacing_samples;
    future_symbol_indices = current_symbol_idx + (1:K) * symbol_spacing_samples;
    
    all_relevant_indices = [past_symbol_indices, current_symbol_idx-half_win:current_symbol_idx+half_win, future_symbol_indices];
    
    if min(all_relevant_indices) >= 1 && max(all_relevant_indices) <= length(rx)
        past_samples = real(rx(past_symbol_indices));
        win_samples = real(rx(current_symbol_idx-half_win:current_symbol_idx+half_win));
        future_samples = real(rx(future_symbol_indices));
        
        features = [past_samples; win_samples; future_samples];
        
        valid_count = valid_count + 1;
        X(valid_count, :) = features(:)';
        Y(valid_count) = symbols(i);
    end
end

X = X(1:valid_count, :);
Y = Y(1:valid_count);

if ~exist('datasets', 'dir')
    mkdir('datasets')
end

tau_str = strrep(num2str(tau), '.', '');
gt_filename = sprintf('datasets/ground_truth_K%d_N%d_tau%s_SNR%d_size%d.csv', K, N, tau_str, SNR_dB, valid_count);
input_filename = sprintf('datasets/input_data_K%d_N%d_tau%s_SNR%d_size%d.csv', K, N, tau_str, SNR_dB, valid_count);

writematrix(Y, gt_filename);
writematrix(X, input_filename);

fprintf('Ground truth saved: %s\n', gt_filename)
fprintf('Input data saved: %s\n', input_filename)
fprintf('Total valid samples: %d\n', valid_count)