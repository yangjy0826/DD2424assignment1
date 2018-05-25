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
    if mod(i, 10) == 0 %This is add for bonus point 1
        GDparams(2)=GDparams(2)*0.9; %This is add for bonus point 1
    end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end