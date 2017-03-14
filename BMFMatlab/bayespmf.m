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

  num_class = 100;
  
  makematrix
  % Update makematrix to load a new data set

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
  %load movieLens
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n');
  
  
  
  
  load pmf_weight
  
  if (size(w1_P1,1) ~= num_p || size(w1_P1,2) ~= num_feat)
      size(w1_P1)
      fprintf(1,'Redoing PMF...\n');
      pause;
      restart = 1;
      pmf
  end
  
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
  %z = repmat(z,num_t,1);
  %Select initial weights of these clusters as an average
  w1_C1_sample = zeros(num_class,num_feat);
  for cc = 1:num_class
      ff = find(z == cc);
      latent_vec = w1_P1_sample(ff,:);
      w1_C1_sample(cc,:) = mean(latent_vec,1);
  end
  
  %Generate transition matrix. Bias to staying in same state.
  Aprior = ones(num_class,num_class); %+ (num_class-2)*eye(num_class)/2;
  A = zeros(num_class);
  for cc = 1:num_class
      A(:,cc) = sampleDirichlet(Aprior(:,cc));
  end
  
  %prior for first node in the HMM chain.
  Azero = sampleDirichlet(ones(num_class,1));
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
       fprintf(1,'movie =%d\r',mm);
       ff = find(count(:,mm)>0);
       MM = w1_P1_sample(ff,:);
       rr = count(ff,mm)-mean_rating;
       covar = inv((alpha_m+beta*MM'*MM));
       mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
       lam = chol(covar); lam=lam'; 
       w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
    end

    %%% Infer posterior distribution over first nodes (Azero)
    %%% (NEW)
    
    nodecounts = zeros(num_class,1);
    for uu = 1:orig_num_p
        nodecounts(z(uu))= nodecounts(z(uu))+1;
    end
    Azero = sampleDirichlet(ones(num_class,1) + nodecounts);
    
    %%% Infer posterior distribution over transition matrix A
    %%% (NEW)
    fprintf(1,'Sampling transition matrix...\r');
    transitioncounts = zeros(num_class)
    for uu = 1:orig_num_p
        zold = z(uu);
        for tt = 2:num_t
            uu = uu + orig_num_p;
            znew = z(uu);
            transitioncounts(zold,znew) = transitioncounts(zold,znew) +1;
            zold = znew;
        end
    end
    for cc = 1:num_class
        Apost = Aprior(cc,:) + transitioncounts(cc,:);
        A(cc,:) = sampleDirichlet(Apost);
    end
    
    count=count';
    %%% Infer posterior distribution over all user-cluster assignments
    %%% (NEW)
    for uu=1:orig_num_p
        fprintf(1,'user =%d\r',uu);
        % select latent movies and ratings for this user
        % need to do some work to account for the user appearing
        % in each time step.
        uf = ((1:num_t) - 1)*orig_num_p + uu;
        ff = [];
        rr = [];
        rt = [];
        for ui = 1:length(uf)
           uuu = uf(ui);
           newf = find(count(:,uuu) > 0);
           ff = [ff;newf];
           rr = [rr; count(newf,uuu)];
           rt = [rt; ui*ones(length(newf),1)];
        end
        MM = w1_M1_sample(ff,:);
        
        %Now we have these vars:
        % uf: list of user-time ids for this user
        % ff: list of movie-ids this user reviewed.
        % MM: Movie latent vecs, ordered by ff.
        % rr: User's rating, ordered by ff.
        % rt: The time of the rating, ordered by ff.
        
        %size(uf)
        %size(ff)
        %size(MM)
        %size(rr)
        %size(rt)
        %uf
        %rt
        
        %begin the forwards-backwards sampling by computing alphas
        alpha = zeros(num_t+1,num_class);
        alpha(1,:) = log(Azero);
        loglikelihoods = ones(num_t,num_class);
        
        for tt = 1:num_t
            %select out observations at this timestep.
            rf = (rt == tt);
            if (sum(rf) > 0)
                
              %fprintf(1,'user %d has a review at time %d!\n',uu,tt);
              %pause;
              fft = ff(rf);
              MMt = MM(rf,:);
              rrt = rr(rf);
              % compute likelihoods of each cluster in generating this user's
              % ratings
              for cc = 1:num_class
                rd = w1_C1_sample(cc,:) * MM' + mean_rating;
                rd = rd' - rr;
                rd = rd.^2;
                % rd now has the squared distance of each rating from its mean.
                % adding these now gives the appropriate scaled likelihood for
                % this cluster
              
                loglikelihoods(tt,cc) = - (beta)^-1*sum(rd);
              end
            end
            
            temp = repmat(alpha(tt,:),num_class,1) + log(A);
            alpha(tt+1,:) = loglikelihoods(tt,:) + log_sum_exp(temp);
            alpha(tt+1,:) = alpha(tt+1,:) - max(alpha(tt+1,:));
        end
        
        %Now we sample backwards
        log_probs = zeros(1,num_class);
        zs_new = zeros(num_t,1);
        for tt=num_t:-1:1
             if (tt == num_t)
                 log_probs = alpha(num_t+1,:)';
             else
                 znext = zs_new(tt+1);
                 log_probs = log(A(:,znext)) + alpha(num_t+1,:)';
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
                   zs_new(tt) = sampleFromDiscrete(probs);
             else
                   zs_new(tt) = randi(num_class,1,1);
                   probs
                   log_probs
                   fprintf(1,'Probabilities exploded.\n');
                   pause; %Shouldn't happen.
             end
        end
        z(uf) = zs_new;
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


