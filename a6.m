clear, clc
N = 100;
sps = 10; % changed this to 10 to make tau*sps an integer
tau = .8;
beta = .3;
span = 6;
SNR = 15;

bits = randi([0 1], 1, N);
symbols = 1-2*bits;

h = rcosdesign(beta, span, sps, 'sqrt');
txsignal = conv(upsample(symbols, tau*sps), h);
rxsignal = awgn(txsignal, SNR, 'measured');
% rxsignal will not be bandlimited

plot(rxsignal), axis tight
grid on, grid minor
xlabel 'Time (t)'
ylabel 'Aplitude (V)'