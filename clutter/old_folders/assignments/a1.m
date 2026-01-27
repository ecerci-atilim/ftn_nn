w = [.5, .3, .5];
for i = 1 : 10
    in = rand(1, 3);
    [decision, wsum] = neuron(in, w);
    fprintf("in   = {%.1f, %.1f, %.1f}\nwsum = %.5f\n", in(1), in(2), in(3), wsum)
    if decision
        fprintf("\tGo for a run! (Neuron fired).\n")
    else
        fprintf("\tStay home. (Neuron did not fire).\n")
    end
end

function [o, wsum] = neuron(in, w)
    wsum = (in*w');
    o = wsum > .5;
end