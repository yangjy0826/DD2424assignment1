function P = EvaluateClassifier(X, W, b)
n = size(X,2);
b = repmat(b,1,n);
s = W*X+b;
P = softmax(s);
end