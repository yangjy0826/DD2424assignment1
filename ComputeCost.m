function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2) + lambda*sum(sum(W.^2));
end