input = 2;
gr_truth = .8;
w = rand;
b = rand;
l_rate = .1;

for i = 1 : 100
    fprintf("Step #%d:\n", i)
    prediction = neuron(input, w, b);
    loss = (prediction-gr_truth)^2;
    fprintf("Prediction: %.5f\nLoss: %.5f\n", prediction, loss)
    
    grad_w = (prediction-gr_truth)*input;
    grad_b = (prediction-gr_truth);
    fprintf("grad_w = %.5f\ngrad_b = %.5f\n", grad_w, grad_b)
    w = w-l_rate*grad_w;
    b = b-l_rate*grad_b;
    fprintf("new w: %.5f\nnew b: %.5f\n\n", w, b)
end

function out = neuron(in, w, b)
    out = in*w + b;
end