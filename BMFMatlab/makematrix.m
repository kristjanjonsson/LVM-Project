% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

%load moviedata
%load movieLens


% Make sure to update these!
num_feat = 30; % Rank 30 decomposition 

% If needed, hard code num_* below instead of taking the max observed.
load ../trainTimed
load ../testTimed
train_vec = trainLabeled;
probe_vec = testLabeled;
max_obs = max(trainLabeled);
num_p = max_obs(1);
num_m = max_obs(2);
num_t = max_obs(4);

%original data, for use only in contexts made aware of time bins.
origtrain = train_vec;
origprobe = probe_vec;
orig_num_p = num_p;
num_p = num_p*num_t;

count = zeros(num_p*num_t,num_m,'single'); %for Netflida data, use sparse matrix instead. 

%HACK: Pretend every time step defines a new user.
if (size(train_vec,2) > 3)
    train_vec = [train_vec(:,1) + (train_vec(:,4)-1)*orig_num_p, train_vec(:,2), train_vec(:,3)];
end

if (size(probe_vec,2) > 3)
    probe_vec = [probe_vec(:,1) + (probe_vec(:,4)-1)*orig_num_p, probe_vec(:,2), probe_vec(:,3)];
end


count = sparse(double(train_vec(:,1)),double(train_vec(:,2)),double(train_vec(:,3)),double(num_p),double(num_m));

count = full(count);

% for mm=1:num_m
%     
%         ff= find(train_vec(:,2)==mm);
%         count(train_vec(ff,1),mm) = train_vec(ff,3);
%     
% end




