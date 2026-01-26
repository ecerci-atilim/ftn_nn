clear, clc
load("data.mat")

N = 5000;
sps = 10;
tau = .8;
beta = .3;
span = 6;
SNR = 15;
window_len = 2*floor(3*sps/2)+1;

Y_categorical = categorical(ydata);

cv = cvpartition(N, 'HoldOut', 0.2);
X_train = xdata(cv.training, :);
Y_train = Y_categorical(cv.training, :);
X_val = xdata(cv.test, :);
Y_val = Y_categorical(cv.test, :);

layers = [
    featureInputLayer(window_len)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ... 
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 30, ...
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

ftn_detector_fnn = trainNetwork(X_train, Y_train, layers, options);

Y_pred = classify(ftn_detector_fnn, X_val);

accuracy = sum(Y_pred == Y_val) / numel(Y_val);