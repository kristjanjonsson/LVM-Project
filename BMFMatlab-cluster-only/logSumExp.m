function result = logSumExp( x )
%LOGSUMEXP Summary of this function goes here
%   Detailed explanation goes here
    m = max(x);
    result = m + log(sum(exp(x-m)));
end