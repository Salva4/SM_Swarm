
% Evaluation of the first part of the functional L (used for the computation of the accuracy)

% Input - X: matrix containing the features
%		  C: matrix of box coordinates
%		  T: size of the data statistic
%		  K: number of discretization boxes

function [gamma] = SPACL_EvaluateGamma_valid(X,C,T,K)

	[~,idx] = min(sqDistance(X, C)');
	gamma = zeros(K,T);
	for k = 1:K
	    gamma(k,find(idx==k)) = 1;
	end
	
end

