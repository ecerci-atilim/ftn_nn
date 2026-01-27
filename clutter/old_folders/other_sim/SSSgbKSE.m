function mesSE = SSSgbKSE(mesin, ISI, gbK)
    s = length(mesin);
    L = length(ISI);
    ssd = zeros(1, s);
    mesSE = zeros(1, s);
    
    for n = 1:s
        % --- SSD: Symbol-by-Symbol Detection ---
        if n == 1
            ssd(n) = mesin(n);
        else
            % Backward ISI cancellation (önceki sembollerden gelen ISI)
            isi_taps = min(n-1, L-1);
            if isi_taps > 0
                ssd(n) = mesin(n) - sign(ssd(n-isi_taps:n-1)) * ISI(isi_taps+1:-1:2)';
            else
                ssd(n) = mesin(n);
            end
        end
        
        % --- Go-back K: Refine past decisions ---
        if n > gbK
            for i = (n - gbK):n
                backward_taps = min(i-1, L-1);
                forward_taps = min(n-i, L-1);
                
                val = mesin(i);
                
                % Forward ISI (gelecek sembollerden - ssd kullan)
                if forward_taps > 0 && i < n
                    val = val - sign(ssd(i+1:i+forward_taps)) * ISI(2:forward_taps+1)';
                end
                
                % Backward ISI (geçmiş sembollerden - mesSE kullan)
                if backward_taps > 0 && i > 1
                    val = val - sign(mesSE(max(1,i-backward_taps):i-1)) * ISI(min(backward_taps+1,L):-1:2)';
                end
                
                mesSE(i) = val;
            end
        end
    end
    
    % Go-back yapılmamış semboller için SSD kullan
    mesSE(mesSE == 0) = ssd(mesSE == 0);
end