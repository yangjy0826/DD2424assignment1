%% Exercise 2.1
clc;
clear;

%1.read data
%Use all the available training data for training (all five batches minus a small
%subset of the training images for a validation set). Decrease the size of the
%validation set down to 1000
[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat'); % training data part1
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat'); % training data part2
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat'); % training data part3
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat'); % training data part4
[X5,Y5,y5] = LoadBatch('data_batch_5.mat'); % training data part5

Xtrain5=X5(:,1:size(X5,2)-1000);
Xtrain=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
Ytrain5=Y5(:,1:size(Y5,2)-1000);
Ytrain=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
ytrain5=y5(:,1:size(X5,2)-1000);
ytrain=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];

Xvalid=X5(:,(size(X5,2)-999):size(X5,2));
Yvalid=Y5(:,(size(Y5,2)-999):size(Y5,2));
yvalid=y5(:,(size(y5,2)-999):size(y5,2));

[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data

%2.initialization
K = 10;
d = 3072;
rng(400);
W = 1/sqrt(10)*randn([K d]); % mean is 0, standard deviation is 0.01
b = 1/sqrt(10)*randn([K 1]);

%3.evaluate
% index=1:5;
% b = repmat(b,1,n);
% P = EvaluateClassifier(Xtrain(:, 1:n), W, b);

%4.cost function
lambda_ = [0.01 0.1 0.2] ; % when lambda = 0, there is no regularization
% J = ComputeCost(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);

%5.accuracy
% acc = ComputeAccuracy(Xtrain(:, 1:n), ytrain(1:n), W, b);

%6.gradients
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtrain(:, index), Ytrain(:,index), W, b, lambda, 1e-6); %Given Gradients
% P = EvaluateClassifier(Xtrain(:, index), W, b);
% [grad_W, grad_b] = ComputeGradients(Xtrain(:, index), Ytrain(:, index), P, W, lambda); %Gradients I compute

%7. mini-batch gradient descent
n_batch = 100; %the number of images in a mini-batch
eta_ = [0.01 0.02 0.2]; %the learning rate
n_epochs = 40; %the number of runs through the whole training set

% Wstar_t=zeros(9,1);
% bstar_t=zeros(9,1);
% loss_t=zeros(9,1);
% loss_v=zeros(9,1);
% accuracy_train=zeros(9,1);
% a=1;
acc_train=0;
for i=1:3
    lambda=lambda_(i);
    for j=1:3
        GDparams = [n_batch, eta_(j), n_epochs];
[Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGD(Xtrain, Ytrain, Xvalid, Yvalid, GDparams, W, b, lambda);

%Calculate the accuracy
accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
if accuracy_train>acc_train %acc_train always store the biggest accuracy
    acc_train=accuracy_train;
    W_max=Wstar_t;
    b_max=bstar_t;
end
    end
end
acc_valid = ComputeAccuracy(Xvalid, yvalid, W_max, b_max);
acc_test = ComputeAccuracy(Xtest, ytest, W_max, b_max);

%% Exercise 2.2
clc;
clear;

%1.read data
%Use all the available training data for training (all five batches minus a small
%subset of the training images for a validation set). Decrease the size of the
%validation set down to 1000
[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat'); % training data part1
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat'); % training data part2
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat'); % training data part3
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat'); % training data part4
[X5,Y5,y5] = LoadBatch('data_batch_5.mat'); % training data part5

Xtrain5=X5(:,1:size(X5,2)-1000);
Xtrain=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
Ytrain5=Y5(:,1:size(Y5,2)-1000);
Ytrain=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
ytrain5=y5(:,1:size(X5,2)-1000);
ytrain=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];

Xvalid=X5(:,(size(X5,2)-999):size(X5,2));
Yvalid=Y5(:,(size(Y5,2)-999):size(Y5,2));
yvalid=y5(:,(size(y5,2)-999):size(y5,2));

[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data

%2.initialization
K = 10;
rng(400);
% 
W = 0.01*randn([K 3072]); % mean is 0, standard deviation is 0.01
b = 0.01*randn([K 1]);

Xtrain_SVM=[Xtrain;ones(1,size(Xtrain,2))];
d = size(Xtrain_SVM,1);
rng(400);
W_SVM=0.01*randn([K d]);
% W_SVM=1/sqrt(10)*randn([K d]);
Xtest_SVM=[Xtest;ones(1,size(Xtest,2))];

%mini-batch
lambda =0; 
n_batch = 100; %the number of images in a mini-batch
eta = 0.001; %the learning rate
n_epochs = 40; %the number of runs through the whole training set
GDparams = [n_batch, eta, n_epochs];
ind=10000;
%cross_entropy
[Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGD(Xtrain(:,1:ind), Ytrain(:,1:ind), Xvalid, Yvalid, GDparams, W, b, lambda); 
accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);

%SVM
[grad_W_SVM,J_SVM] = ComputeSVMGradients(Xtrain_SVM(:,1:ind), ytrain(1:ind), W_SVM, lambda);
Wstar_t_SVM = SVM_MiniBatchGD(Xtrain_SVM(:,1:ind), ytrain(1:ind), GDparams, W_SVM, lambda);
accuracy_test_SVM = ComputeAccuracySVM(Xtest_SVM(:,1:ind), ytest(1:ind), Wstar_t_SVM);

%% functions for SVM
function acc = ComputeAccuracySVM(X, y, W)
P = W*X;
[~,n] = size(y);
correct = 0;
for i = 1:n
    [~,k(i)] = max(P(:,i));
    if y(i)+1 == k(i)
          correct = correct+1;
    end
end
acc = correct/n;
end

function [grad_W,J] = ComputeSVMGradients(X, y, W, lambda)
num_train=size(X,2);
num_classes=size(W,1);
loss=0;
grad_W=zeros(size(W,1),size(W,2));
for i=1:num_train
    scores=W*X(:,i);
    correct_score=scores(y(i)+1);
    for j=1:num_classes
        if j==y(i)+1
            continue
        end
        margin=scores(j)-correct_score+1; % note delta = 1
        if margin>0
            loss =loss+margin;
%             grad_W(:,y(i)+1) = grad_W(:,y(i)+1)-X(i);
%             grad_W(:, j) = grad_W(:, j)+X(i);
            grad_W(y(i)+1,:) = grad_W(y(i)+1,:)-X(:,i)';
            grad_W(j,:) = grad_W(j,:)+X(:,i)';
        end
    end
end
J=loss/num_train+lambda*sum(sum(W.^2));
grad_W=grad_W/num_train+2*lambda*W;
end

function Wstar = SVM_MiniBatchGD(X, y, GDparams, W, lambda)
N = size(X,2);
for i=1:GDparams(3)
    for j=1:N/GDparams(1)
        j_start = (j-1)*GDparams(1) + 1;
        j_end = j*GDparams(1);
        % inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        ybatch = y(j_start:j_end);

        [grad_W, ~] = ComputeSVMGradients(Xbatch, ybatch,W, lambda);

        W = W - GDparams(2) * grad_W;
%     end
%     [~,J(i)] = ComputeSVMGradients(X, y, W, lambda);
%     if mod(i, 10) == 0 %This is add for bonus point 1 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.9; %This is add for bonus point 1
%     end %This is add for bonus point 1 %This is add for bonus point 1
    end
Wstar = W;
end
%% functions for cross-entropy
function P = EvaluateClassifier(X, W, b)
n = size(X,2);
b = repmat(b,1,n);
s = W*X+b;
P = softmax(s);
end

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2) + lambda*sum(sum(W.^2));
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~,n] = size(y);
correct = 0;
for i = 1:n
    [~,k(i)] = max(P(:,i));
    if y(i)+1 == k(i)
          correct = correct+1;
    end
end
acc = correct/n;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
g = -(Y-P)';
grad_W = g'*X';
grad_b = sum(g,1)';
[~,n]=size(X);
grad_W=grad_W/n+2*lambda*W;
grad_b=grad_b/n;
end

function [Wstar, bstar, J, J2] = MiniBatchGD(X, Y,X2,Y2, GDparams, W, b, lambda)
N = size(X,2);
for i=1:GDparams(3)
    for j=1:N/GDparams(1)
        j_start = (j-1)*GDparams(1) + 1;
        j_end = j*GDparams(1);
        % inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);

        W = W - GDparams(2) * grad_W;
        b = b - GDparams(2) * grad_b;
    end
    J(i) = ComputeCost(X, Y, W, b, lambda);
    J2(i) = ComputeCost(X2, Y2, W, b, lambda);
%     if mod(i, 10) == 0 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.95; %This is add for bonus point 1
%     end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end
%% functions for both methods
function [X, Y, y] = LoadBatch(filename)
 A = load(filename);
 X = double(A.data)/double(255); %normalized to figures between 0 and 1
 %X is of type "double"
 y = A.labels;
 [a,~] = size(y);
 K = 10;
 Y = zeros(a,K);
 for i = 1:a
 Y(i,y(i)+1) = 1;
 end
 X = X';
 Y = Y';
 y = y';
end