inputs = rand(1, 3);
weights = rand(3, 3);
biases = rand(1, 3);

fprintf("inputs:  %s\n", sprintf("%.2f ", inputs))
fprintf("biases:  %s\n", sprintf("%.2f ", biases))
fprintf("weights:\n")
disp(weights)

wsum = neuron(inputs, weights, biases);
[ro, so] = activate_layer(wsum);

fprintf("ReLu:    %s\n", sprintf("%.5f ", ro))
fprintf("Sigmoid: %s\n", sprintf("%.5f ", so))

function wsum =  neuron(in, w, b)
    wsum = in*w' + b;
end

function [relu, sigm] = activate_layer(wsum)
    relu = max(0, wsum);
    sigm = 1./(1+exp(-wsum));
end