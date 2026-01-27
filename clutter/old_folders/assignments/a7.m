clear, clc
N = 5000;
sps = 10; % changed this to 10 to make tau*sps an integer
tau = .8;
beta = .3;
span = 6;
SNR = 15;
window_len = 2*floor(3*sps/2)+1;
pulse_loc = span*sps+1;
xdata = zeros(N, window_len);
ydata = zeros(N, 1);
total_delay = floor(window_len/2);

bits = randi([0 1], 1, N);
symbols = 1-2*bits;

h = rcosdesign(beta, span, sps, 'sqrt');
txsignal = conv(upsample(symbols, tau*sps), h);
rxsignal = awgn(txsignal, SNR, 'measured');
rxMF = conv(rxsignal, h); % matched filter to bandlimit

ploc = pulse_loc;
for i = 1 : N
    xdata(i, :) = rxMF(ploc-total_delay : ploc+total_delay);
    ydata(i) = bits(i);
    ploc = ploc + sps*tau;
end

save("data.mat", 'xdata', 'ydata')