clear, clc
rng(0)
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

N = 11;
tau_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
rolloff = .3;
gdelay = 4;
fs = 10;
fd = 1;
sps = fs/fd;
span = 2*gdelay;

h = rcosdesign(rolloff, span, sps, 'sqrt');

colors = cool(length(tau_values));

for i = 1 : 2
    if i == 1
        b = [1 1 0 1 0 0 1 0 0 1 0];
    else
        b = [1 1 0 1 0 1 1 0 0 1 0];
    end
    m = 1-2*b;
    
    subplot(2, 1, i)
    hold on
    grid on
    grid minor
    
    h_sig = gobjects(length(tau_values), 1);  % Sinyal handle'ları
    h_around = [];  % Samples around symbol (sadece bir tane)
    h_symbol = [];  % Symbol (sadece bir tane)
    
    for k = 1:length(tau_values)
        tau = tau_values(k);
        ploc = 2*gdelay*fs + 1 + floor(N/2)*sps*tau;
        
        txus = upsample(m, tau*sps);
        txsig = conv(txus, h);
        rxmf = conv(txsig, h);
        
        % Sinyal çizimi
        h_sig(k) = plot(rxmf, 'Color', colors(k,:), 'LineWidth', 1.2);
        
        % Samples around symbol ve Symbol - sadece ilk tau için legend'a ekle
        pdom = round(ploc-tau*sps/2):round(ploc+tau*sps/2);
        pran = rxmf(pdom);
        
        if k == 1
            h_around = plot(pdom, pran, 'r', 'LineWidth', 2);
            h_symbol = plot(ploc, rxmf(ploc), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
        else
            plot(pdom, pran, 'r', 'HandleVisibility', 'off', 'LineWidth', 2);
            plot(ploc, rxmf(ploc), 'g*', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end
    end
    
    % Legend oluştur
    leg_handles = [h_sig; h_around; h_symbol];
    leg_labels = [arrayfun(@(t) sprintf('$\\tau = %.1f$', t), tau_values, 'UniformOutput', false), ...
                  {'Samples around symbol', 'Symbol'}];
    legend(leg_handles, leg_labels, 'Location', 'se')
    
    xlabel('Samples')
    ylabel('Amplitude')
    mid = floor(N/2) + 1;
    str_parts = arrayfun(@(x) sprintf('%d', x), b, 'UniformOutput', false);
    str_parts{mid} = sprintf('\\textbf{%d}', b(mid));
    title(strjoin(str_parts, ''))
    axis tight
end
subplot 211
xlim([50 200])
subplot 212
xlim([50 200])