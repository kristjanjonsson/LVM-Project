function s = sampleDirichlet(dirParams)
    sample = gamrnd(dirParams,ones(size(dirParams)));
    sample = sample ./ sum(sample);
    s = sample;
end