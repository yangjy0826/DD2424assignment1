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
