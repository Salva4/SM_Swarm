
% Computation of the squared distance

% Input - X and Y: matrices with coherent dimensions

function D = sqDistance(X, Y)
D = bsxfun(@plus, dot(X,X,1)' , dot(Y,Y,1)) - 2*(X'*Y);
