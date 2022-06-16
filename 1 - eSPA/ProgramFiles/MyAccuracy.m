function [ KL ] = MyAccuracy(x,y)

	[~,ix] = max(x);
	[~,iy] = max(y);
	KL = -sum(ix==iy) * size(x,1);

end

