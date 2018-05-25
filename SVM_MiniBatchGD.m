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
%         GDparams(2)=GDparams(2)*0.95; %This is add for bonus point 1
%     end %This is add for bonus point 1 %This is add for bonus point 1
    end
Wstar = W;
end