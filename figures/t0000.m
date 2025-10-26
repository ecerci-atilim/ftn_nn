clear, clc
rng(0)

set(groot,'defaultAxesTickLabelInterpreter','latex');      % Interpreter definition for axes ticks of figures
set(groot,'defaulttextinterpreter','latex');               % Interpreter definition for default strings casted on figures
set(groot,'defaultLegendInterpreter','latex');             % Interpreter definitions for default legend strings displayed on figures

N = 11;
tau = .8;

rolloff = .3;
gdelay = 4;
fs = 10;
fd = 1;
sps = fs/fd;
span = 2*gdelay;

ploc = 2*gdelay*fs + 1 + floor(N/2)*sps*tau;

h = rcosdesign(rolloff, span, sps, 'sqrt');
hh = conv(h, h);

for i = 1 : 2
    % b = [randi([0, 1], 1, floor(N/2)), 1, randi([0, 1], 1, floor(N/2))];
    if i == 1
        b = [0 0 0 1 0 1 1 1 1 1 1];
    else
        b = [0 0 0 1 0 0 1 1 1 1 1];
    end
    m = 1-2*b;
    txus = upsample(m, tau*sps);
    txsig = conv(txus, h);
    rxmf = conv(txsig, h);

    subplot(2, 1, i)
    plot(rxmf)
    axis tight
    hold on
    grid on
    grid minor
    xlabel Samples
    ylabel Amplitude
    title(sprintf("%d",b))
    
    pdom = ploc-tau*sps/2:ploc+tau*sps/2;
    pran = rxmf(pdom);
    plot(pdom, pran, 'Color', 'r')
    plot(ploc, rxmf(ploc), 'g*')

    legend('Signal', 'Samples around symbol', 'Symbol')
    xlim([50 200])

end