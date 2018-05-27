clc;
clear;

%1.read data
[Xtrain,Ytrain,ytrain] = LoadBatch('data_batch_1.mat'); % training data
[Xvalid,Yvalid,yvalid] = LoadBatch('data_batch_2.mat'); % validation data
[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data

%2.initialization
K = 10;
d = 3072;
rng(400);
W = 0.01*randn([K d]); % mean is 0, standard deviation is 0.01
b = 0.01*randn([K 1]);

%3.evaluate
index=1:5;
% b = repmat(b,1,n);
% P = EvaluateClassifier(Xtrain(:, 1:n), W, b);

%4.cost function
lambda = 1; % when lambda = 0, there is no regularization
% J = ComputeCost(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);

%5.accuracy
% acc = ComputeAccuracy(Xtrain(:, 1:n), ytrain(1:n), W, b);

%6.gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtrain(:, index), Ytrain(:,index), W, b, lambda, 1e-6); %Given Gradients
P = EvaluateClassifier(Xtrain(:, index), W, b);
[grad_W, grad_b] = ComputeGradients(Xtrain(:, index), Ytrain(:, index), P, W, lambda); %Gradients I compute
dif_b=max(max(ngrad_b-grad_b));
dif_W=max(max(ngrad_W-grad_W));

%7. mini-batch gradient descent
n_batch = 100; %the number of images in a mini-batch
eta = 0.01; %the learning rate
n_epochs = 40; %the number of runs through the whole training set
GDparams = [n_batch, eta, n_epochs];
[Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGD(Xtrain, Ytrain, Xvalid, Yvalid, GDparams, W, b, lambda);

%Calculate the accuracy
accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
acc_valid = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);

%Draw loss picture
figure();
plot(loss_t);
hold on;
plot(loss_v);
grid on;
legend('training loss','validation loss');
xlabel('epoch');
ylabel('loss');
hold off;

% visualization the weight matrix
for i=1:10
im = reshape(Wstar_t(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure();
imshow(cell2mat(s_im));
