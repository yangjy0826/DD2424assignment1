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