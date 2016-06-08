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

rand('state',0);
randn('state',0);

if restart==1 
  restart=0; 
  epoch=1; 
  maxepoch=50;

  iter=0; 
  num_m = 1682;
  num_p = 943;
  num_feat = 10;
  num_class = 10;

  % Initialize hierarchical priors 
  beta=2; % observation noise (precision) 
  mu_u = zeros(num_feat,1);
  mu_m = zeros(num_feat,1);
  alpha_u = eye(num_feat);
  alpha_m = eye(num_feat);  

  % parameters of Inv-Whishart distribution (see paper for details) 
  WI_u = eye(num_feat);
  b0_u = 2;
  df_u = num_feat;
  mu0_u = zeros(num_feat,1);

  WI_m = eye(num_feat);
  b0_m = 2;
  df_m = num_feat;
  mu0_m = zeros(num_feat,1);

  %load moviedata
  load movieLens
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
  makematrix

  load pmf_weight
  err_test = cell(maxepoch,1);

  w1_P1_sample = w1_P1; 
  w1_M1_sample = w1_M1; 
  clear w1_P1 w1_M1;

  % Initialization using MAP solution found by PMF. 
  %% Do simple fit
  mu_u = mean(w1_P1_sample)';
  d=num_feat;
  alpha_u = inv(cov(w1_P1_sample));

  mu_m = mean(w1_M1_sample)';
  alpha_m = inv(cov(w1_P1_sample));

  count=count';
  probe_rat_all = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
  counter_prob=1;
  
  %Initialize parameters for latent assignments
  alpha = ones(num_class,1); %prior for theta
  theta = sampleDirichlet(alpha); %distribution of assignments
  z = randi(num_class,num_p,1); %latent assignments
  %Select initial weights of these clusters as an average
  w1_C1_sample = zeros(num_class,num_feat);
  for cc = 1:num_class
      ff = find(z == cc);
      latent_vec = w1_P1_sample(ff,:);
      w1_C1_sample(cc,:) = mean(latent_vec,1);
  end
%   randind = randperm(num_p,num_class);
%   w1_C1_sample = w1_P1_sample(randind,:);

end


for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  N = size(w1_M1_sample,1);
  x_bar = mean(w1_M1_sample)'; 
  S_bar = cov(w1_M1_sample); 

  WI_post = inv(inv(WI_m) + N/1*S_bar + ...
            N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
  WI_post = (WI_post + WI_post')/2;

  df_mpost = df_m+N;
  alpha_m = wishrnd(WI_post,df_mpost);   
  mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
  lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
  mu_m = lam*randn(num_feat,1)+mu_temp;


  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  %%%
  %%% No Modification? Hyperparameter over cluster means/covariances
  %%% But now we have fewer clusters than users. However, clusters
  %%% should be weighted by their user counts, which degrades back
  %%% to these original sums over all users.
  N = size(w1_P1_sample,1);
  x_bar = mean(w1_P1_sample)';
  S_bar = cov(w1_P1_sample);

  WI_post = inv(inv(WI_u) + N/1*S_bar + ...
            N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
  WI_post = (WI_post + WI_post')/2;
  df_mpost = df_u+N;
  alpha_u = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
  lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  
  %
  % No modifications for movie feature sampling.
  % User sampling modified to sample clusters instead.

  for gibbs=1:2 
    fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    %%%
    %%% No modifications for movie feature sampling.
    count=count';
    for mm=1:num_m
       % fprintf(1,'movie =%d\r',mm);
       ff = find(count(:,mm)>0);
       MM = w1_P1_sample(ff,:);
       rr = count(ff,mm)-mean_rating;
       covar = inv((alpha_m+beta*MM'*MM));
       mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
       lam = chol(covar); lam=lam'; 
       w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
    end

    %%% Infer posterior distribution over all latent assignment
    %%% distribution.
    %%% (NEW)
    % Basic dirichlet update: Given observed assignments, generate
    % new proportions.
    classcount = zeros(num_class,1);
    for cc = 1:num_class
        classcount(cc) = sum(z == cc);
    end
    alpha_post = alpha + classcount;
    theta = sampleDirichlet(alpha_post);
    
    
    count=count';
    %%% Infer posterior distribution over all user-cluster assignments
    %%% (NEW)
    for uu=1:num_p
        % fprintf(1,'user =%d\r',uu);
        % select latent movies and ratings for this user
        ff = find(count(:,uu)>0);
        MM = w1_M1_sample(ff,:);
        rr = count(ff,uu);
        % add in influence of the prior theta 
        log_probs = log(theta);
        % compute likelihoods of each cluster in generating this user's
        % ratings
        for cc = 1:num_class
            rd = w1_C1_sample(cc,:) * MM' + mean_rating;
            rd = rd' - rr;
            rd = rd.^2;
            % rd now has the squared distance of each rating from its mean.
            % adding these now gives the appropriate scaled likelihood for
            % this cluster
            
            log_probs(cc) = log_probs(cc) - (2*beta)^-1*sum(rd);
        end
        %renormalize
        max_log = max(log_probs);
        log_probs = log_probs - max_log;
        log_norm = log_sum_exp(log_probs);
        
        probs = exp(log_probs - log_norm );
        %DEBUG:
        %   fprintf(1,'sumprob =%d\r',sum(probs));
        %sample
        if (sum(probs) > 0)
            z(uu) = sampleFromDiscrete(probs);
        else
            z(uu) = randi(num_class,1,1);
            pause; %Shouldn't happen.
        end
    end
        
    
    
    %%% Infer posterior distribution over all user-cluster feature vectors 
    %%%
    %%% Mild modification: Instead of selecting all movies/ratings for one
    %%% user, select all movies/ratings for a whole cluster of users.
     for cc=1:num_class
       fprintf(1,'user-cluster  =%d\r',cc);
       % create  multiset of movies rated by users in this cluster.
       % For each user in cluster, generate list of movies and ratings
       % that are nonzero
       uf = find(z == cc);
       if (sum(uf) == 0)
           continue
       end
       ff = [];
       rr = [];
       for ui = 1:length(uf)
           uu = uf(ui);
           newf = find(count(:,uu) > 0);
           ff = [ff;newf];
           rr = [rr; count(newf,uu)];
       end
       MM = w1_M1_sample(ff,:);
       rr = rr-mean_rating;
       % At this point MM has a collection of movie latent vecs,
       % and rr has the rating given for each row of MM.
       % When a movie was rated by multiple users of this cluster,
       % that movie latent vec is duplicated in MM.
       % When ratings are omitted, they do not show up in rr or MM.
       %
       % The following is essentially now unmodified. Given all these
       % ratings and movie vecs, compute a posterior for the cluster vec.
       covar = inv((alpha_u+beta*(MM'*MM)));
       mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
       lam = chol(covar); lam=lam';
       new_latent_vec = lam*randn(num_feat,1)+mean_u;
       w1_C1_sample(cc,:) = new_latent_vec;
       % HACK: rather than update all the other code to be
       % user-cluster aware, we'll just hard assign each user
       % to the latent representation of their cluster.
       w1_P1_sample(uf,:) = repmat(new_latent_vec',length(uf),1);
     end
   end 

   probe_rat = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   counter_prob=counter_prob+1;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   iter=iter+1;
   overall_err(iter)=err;

  fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);

end 

