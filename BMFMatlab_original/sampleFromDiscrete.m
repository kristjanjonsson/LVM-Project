function s = sampleFromDiscrete(probs)
    temp = rand();
    total = 0;
    for i = 1:length(probs)
        total = total + probs(i);
        if temp < total
            s = i;
            return;
        end
    end
    error('exhausted...');
end