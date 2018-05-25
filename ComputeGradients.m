function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
g = -(Y-P)';
grad_W = g'*X';
grad_b = sum(g,1)';
[~,n]=size(X);
grad_W=grad_W/n+2*lambda*W;
grad_b=grad_b/n;
end