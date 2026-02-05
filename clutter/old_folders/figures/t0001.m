clear, clc
rng(17)

set(groot,'defaultAxesTickLabelInterpreter','latex');      % Interpreter definition for axes ticks of figures
set(groot,'defaulttextinterpreter','latex');               % Interpreter definition for default strings casted on figures
set(groot,'defaultLegendInterpreter','latex');             % Interpreter definitions for default legend strings displayed on figures

N = 11;
tau = .8;

rolloff = .3;
gdelay = 4;
fs = 20;
fd = 1;
sps = fs/fd;
span = 2*gdelay;

ploc = 2*gdelay*fs + 1 + floor(N/2)*sps*tau;

h = rcosdesign(rolloff, span, sps, 'sqrt');
hh = conv(h, h);

for i = 1 : 2
    b = [randi([0, 1], 1, floor(N/2)), 1, randi([0, 1], 1, floor(N/2))];
    m = 1-2*b;
    txus = upsample(m, tau*sps);
    txsig = conv(txus, h);
    rxmf = conv(txsig, h);

    subplot(2, 1, i)
    plot(rxmf)
    axis tight
    hold on
    grid on
    % grid minor
    xlabel '\"{O}rnekler'
    ylabel Genlik
    title(sprintf("%d",b))
    
    pdom = ploc-10:ploc+10;
    % pdom = [pdom, 
    pran = rxmf(pdom);
    plot(pdom, pran, 'r.', 'MarkerSize', 8)
    pdom = [ploc-4*tau*sps, ploc-3*tau*sps, ploc-2*tau*sps, ploc-1*tau*sps,...
              ploc+4*tau*sps, ploc+3*tau*sps, ploc+2*tau*sps, ploc+1*tau*sps];
    pran = rxmf(pdom);
    plot(pdom, pran, 'ro')
    plot(ploc, rxmf(ploc), 'g.', 'MarkerSize', 15)

    legend('Sinyal', 'Etraftaki \"{O}rnekler', 'Komşu Semboller', 'Sembol')
    xlim(2*[50 200])

end