function [estimate, varargout] = gfmm( data, D, logF, expF, varargin );
% GFMM Estimates a Geodesic Finite Mixture Model from data on a known manifold.
%
% Copyright (C) <2014> <Edgar Simo-Serra>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the version 3 of the GNU General Public License
% as published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% General Public License for more details.      
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
%
% Edgar Simo-Serra, Institut de Robotica i Informatica Industrial (CSIC/UPC), September 2014.
% esimo@iri.upc.edu, http://www-iri.upc.es/people/esimo/
%
% SYNOPSIS [mix,info] = gfmm( data, D, LogF, ExpF, ... )
%
% INPUT data: Input data on the manifold
%       D: Dimension of the manifold
%       logF: Logarithm map function for the manifold.
%       expF: Exponential map function for the manifold.
%       ...: Optional parameters, see PARAMETERS for a full overview.
%
% PARAMETERS
%     maxloops: Maximum number of iterations to do (default 500)
%     Cmax: Initial number of cluster components (default ceil(min(50, N/(D*D)/3)))
%     Cmin: Minimum number of cluster components (default 1)
%     verbose: Execution verbosity (default 0)
%     thr: Likelyhood convergence threshold (default 1e-6)
%     animate: Whether or not to show animations (default 0)
%     covtype: Type of covariance. 0 is full, 1 is diagonal and a custom matrix is a mask. (default 0)
%     logging: Whether or not to enable logging (default 0)
%     logfile: File to log to (default 'logging.mat')
%     seqiter: How many iterations to do sequentially instead of block updates. (default inf)
%     meantol: Tolerance when calculating geodesic mean (default 1e-10)
%     covinit: Initial scaling of the covariance. (default 0)
%     meanwthresh: Mass to use when calculating geodesic mean. (default 0.9999)
%
% OUTPUT mix: The estimated Geodesic Finite Mixture Model
%        info: Additional information about each iteration (optional)
%
% REMARKS
%  If you use this code please cite [1] as:
%
%  @InProceedings{SimoSerraBMVC2014,
%     author = {Edgar Simo-Serra and Carme Torras and Francesc Moreno-Noguer},
%     title = {{Geodesic Finite Mixture Models}},
%     booktitle = "Proceedings of the British Machine Vision Conference (BMVC)",
%     year = 2014,
%   }
%
% References:
%   [1] E. Simo-Serra, C. Torras, and F. Moreno-Noguer.
%   Geodesic Finite Mixture Models. BMVC 2014.
%   [2] M. Figueiredo, and A. Jain.
%   Unsupervised Learning on Finite Mixture Models.
%   PAMI 24(3):381–396, 2002.
%
% This code implements [1] and is directly based on [2] and code published on Figueiredo homepage: http://www.lx.it.pt/~mtf/
%
% SEE ALSO gfmm_sample
%
% Author: Edgar Simo-Serra
% Date: 08-Sep-2014
% Version: 1.0

[N, DD]  = size(data);  % number of points (n), dimensions (d)

% defaults
conf = struct(...
   'maxloops', 500, ...
   'Cmax',     ceil(min(50, N/(D*D)/3)), ...
   'Cmin',     1, ...
   'verbose',  0, ...
   'thr',      1e-6, ...
   'animate',  0, ...
   'covtype',  0, ...
   'logging',  0, ...
   'logfile',  'logging.mat', ...
   'seqiter',  inf, ..., % off by default
   'meantol',  1e-10, ...
   'covinit',  0, ...
   'meanwthresh', 0.9999 ...
   );

if nargout>1
   conf.logging = 1;
   varargout{1} = [];
end

conf  = getargs(conf, varargin);
C     = conf.Cmax;

% If no output struct to store stuff in, disable logging
if nargout<2
   conf.logging = 0;
end

% for logging
log_covfixer2  = {};
log_loglikes   = {};
log_C          = {};
log_costs      = {};
log_annih      = {};
log_initialmix = {};
log_mixtures   = {};
log_Cids       = {};

% The number of free parameters in a Gaussian
Nparc    = D; % Means, they are defined on the manifold, NOT the embedding space
if conf.covtype == 0;
   Nparc = Nparc + D*(D+1)/2; % (N)
elseif conf.covtype == 1;
   Nparc = Nparc + D;   % (N)
else
   % Using a covariance mask
   assert( all(size(conf.covtype)==[D,D]) );
   Nparc = Nparc + sum(conf.covtype(:));
end

Nparc2   = Nparc/2;
N_limit  = (Nparc+1)*3*conf.Cmin;
if N < N_limit
   warning_wrap('gfmm:data_amount', ...
      ['Training data may be insufficient for selected ' ...
      'minimum number of components. ' ...
      'Have: ' num2str(N) ', recommended: >' num2str(N_limit) ...
      ' points.']);
end

% This is trick, but how it works is that the tangent planes are defined in
% R^3 while the actual covariance is defined on the tangent plane to SO(3) which
% is actually R^2 representation of SO(2)
if (C<1) | (C>N)
   C  = N;
   mu = data;
else
   % initialize mu as random points from data
   permi = randperm(N);
   mu  = data( permi(1:C),: );
end
mu = mu.';

% initialize sigma by for each mu projecting all the data points onto
% the tangent plane and trying to estimate a covariance distribution on that
alpha = ones(1,C) * (1/C);
Ldata = zeros( N, D );
sigma = zeros( D, D, C );
u     = zeros(N,C);   % semi_indic.'
Cids  = 1:C;
tbest = -1;
tinit = tic();
for c = 1:C;
   Ldata = logF( data, mu(:,c) );

   %s2          = max(diag(gmmb_covfixer(cov(Ldata,1))/10)); % originally /10
   %sigma(:,:,c)   = s2*eye(D);
   %sigma(:,:,c)   = diag(mean(Ldata(:,:,c) .* conj(Ldata(:,:,c)))) / 10;
   if conf.covinit==0;
      sigma(:,:,c)   = diag(mean(Ldata .* conj(Ldata))) / sqrt(C);
   else
      sigma(:,:,c)   = diag(mean(Ldata .* conj(Ldata))) / conf.covinit;
   end

   % Calculate u
   u(:,c)       = cmvnpdf_clipped( mu(:,c), Ldata, sigma(:,:,c) );
end
indic = u .* repmat(alpha, N,1);
if conf.verbose > 0;
   fprintf( 'Initial distributions set up in %.1f seconds\n', toc(tinit) );
end
if conf.animate > 0;
   h_history = figure();
end

log_initialmix = struct(...
   'weight',   alpha, ...
   'mu',      mu, ...
   'sigma',   sigma);

t     = 0;
Cnz   = C;  % (k_nz) k = kmax
Lmin  = NaN;

   % Fancy drawings
if conf.animate ~= 0
   aniH = my_plot_init;
   my_plot_ellipses( aniH, logF, data, mu, sigma, alpha, u );
end

[ old_loglike, old_L ] = compute_L( indic, alpha, Nparc2, Cnz, N );
old_loglike2 = old_loglike;

% Loop until minimum cluster amount is reached
while Cnz >= conf.Cmin
   repeating     = 1;

   % The optimization is done in the repeating cycle here
   fixing_cycles  = 0;
   loops        = 0; % We reset the loops for each destruction
   tloop        = tic();
   while repeating
      t        = t+1;
      loops    = loops +1;
      tinner   = tic();
  
      % Handle the updating algorithm
      if t < conf.seqiter;
         [ alpha, mu, sigma, u, log_fixcount, annihilated_count, Cids ] = ...
            update_sequential( logF, expF, data, D, ...
                  alpha, mu, sigma, u, Nparc2, Cids, conf );
      else
         [ alpha, mu, sigma, u, log_fixcount, annihilated_count, Cids ] = ...
            update_block( logF, expF, data, D, ...
                  alpha, mu, sigma, u, Nparc2, Cids, conf );
      end
      C     = size(mu,2);
      Cnz   = C;

      fixed_on_this_round = (log_fixcount > 0);

      if conf.logging>0
        % the component indexes are not constants,
        % cannot record component-wise fix counts
        log_covfixer2{t,1} = log_fixcount;
      end

      if conf.animate ~= 0
         my_plot_ellipses( aniH, logF, data, mu, sigma, alpha, u );
      end
   
      % Updates membership probability of samples for next iteration
      [ u, indic ] = update_u( logF, N, data, mu, sigma, alpha );

      % Update likelyhood
      [ loglike, L ] = compute_L( indic, alpha, Nparc2, Cnz, N );

      % Display partial results
      if conf.verbose ~= 0
         fprintf( 'Cnz=%d, t=%d => %.3e <? %.3e  (L=%.3e) in %.1f secs\n', Cnz, t, ...
               abs(loglike - old_loglike), conf.thr*abs(old_loglike), L, toc(tinner) );
      end
      
      % Log the information
      log_loglikes{t} = loglike;
      log_C{t}       = Cnz;
      log_costs{t}   = L;
      log_annih{t}   = [annihilated_count, 0];
      if conf.animate > 0;
         figure( h_history );
         subplot( 2, 1, 1 ); grid on;
         plot( 1:t, cell2mat(log_costs) );
         subplot( 2, 1, 2 ); grid on;
         plot( 1:t, cell2mat(log_C) );
      end
      if conf.logging>0

         % More detailed logging
         if conf.logging>1
            log_mixtures{t}   = struct(...
               'weight',   alpha, ...
               'mu',       mu, ...
               'sigma',    sigma);
            log_Cids{t}       = Cids;

            pga_logging = struct(...
               'iterations',     {t}, ...
               'costs',          {cat(1,log_costs{:})}, ...
               'annihilations',  {sparse(cat(1,log_annih{:}))}, ...
               'covfixer2',      {cat(1,log_covfixer2{:})}, ...
               'loglikes',       {cat(1,log_loglikes{:})}, ...
               'initialmix',     {log_initialmix}, ...
               'mixtures',       {log_mixtures}, ...
               'cluster_ids',    {log_Cids}, ...
               'best',           tbest );
            save( conf.logfile, '-v7.3', 'pga_logging', 'conf' );
         end
      end

      if fixed_on_this_round ~= 0
         % if any cov's were fixed, increase count and
         % do not evaluate stopping threshold.
         fixing_cycles = fixing_cycles +1;
         if conf.verbose ~= 0
            disp(['fix cycle ' num2str(fixing_cycles)]);
         end
      else
         % no cov's were fixed this round, reset the counter
         % and evaluate threshold.
         fixing_cycles = 0;
         %if (abs(loglike/old_loglike - 1) < conf.thr)
         % We store two offset means of 2 iterations to low-pass a bit
         ll_cur = (loglike+old_loglike)/2;
         ll_old = (old_loglike+old_loglike2)/2;
         if (abs(ll_cur/ll_old - 1) < conf.thr)
            repeating = 0;
         end
      end
  
      % Store old values
      old_L      = L;
      old_loglike2 = old_loglike;
      old_loglike = loglike;
      
      if fixing_cycles > 20
         repeating = 0;
      end
      if loops > conf.maxloops
         repeating = 0;
      end
   end % while repeating
   if conf.verbose ~= 0
      fprintf( 'Cnz = %d in %.1f secs\n', Cnz, toc(tloop) );
   end

   % Store best
   if isnan(Lmin) || (L <= Lmin)
      tbest = t;
      Lmin  = L;
      estimate = struct('mu', mu,...
         'sigma', sigma,...
         'weight', alpha.');
   end
      

   % annihilate the least probable component
   m         = find(alpha == min(alpha(alpha>0)));
   alpha(m(1)) = 0;
   Cnz       = Cnz -1;
   % alpha doesn't need to be normalized here, even if it would seem logical to do so.
   
   if conf.logging > 0
      log_annih{t}(2) = 1;
   end
   
   if Cnz > 0
      alpha = alpha / sum(alpha);
   
      % purge alpha == 0 if necessary
      if length(find(alpha==0)) > 0
         nz    = find(alpha>0);
         alpha = alpha(nz);
         mu    = mu(:,nz);
         sigma = sigma(:,:,nz);
         u     = u(:,nz);
         C     = length(nz);
         Cids  = Cids(nz);
      end
      
      % Recalculate cluster probability
      [ u, indic ] = update_u( logF, N, data, mu, sigma, alpha );
    
      % Calculate likelyhoods
      [ old_loglike, old_L ] = compute_L( indic, alpha, Nparc2, Cnz, N );
   end
end

if conf.logging>1
   varargout{1} = struct(...
      'iterations',     {t}, ...
      'costs',          {cat(1,log_costs{:})}, ...
      'annihilations',  {sparse(cat(1,log_annih{:}))}, ...
      'covfixer2',      {cat(1,log_covfixer2{:})}, ...
      'loglikes',       {cat(1,log_loglikes{:})}, ...
      'components',     {cat(1,log_C{:})}, ...
      'initialmix',     {log_initialmix}, ...
      'mixtures',       {log_mixtures}, ...
      'cluster_ids',    {log_Cids}, ...
      'best',           tbest ...
      );
end
if conf.logging == 1
   varargout{1} = struct(...
      'iterations',     {t}, ...
      'costs',          {cat(1,log_costs{:})}, ...
      'annihilations',  {sparse(cat(1,log_annih{:}))}, ...
      'covfixer2',      {cat(1,log_covfixer2{:})}, ...
      'loglikes',       {cat(1,log_loglikes{:})}, ...
      'components',     {cat(1,log_C{:})}, ...
      'cluster_ids',    {log_Cids}, ...
      'best',           tbest ...
      );
end

% purge alpha==0
e              = estimate;
inds           = find(e.weight>0);
estimate.mu    = e.mu(:,inds);
estimate.sigma = e.sigma(:,:,inds);
estimate.weight = e.weight(inds);

if conf.animate ~= 0
   [ u, indic ] = update_u( logF, N, data, estimate.mu, estimate.sigma, estimate.weight' );
   my_plot_ellipses( aniH, logF, data, estimate.mu, estimate.sigma, estimate.weight', u );
end

if conf.verbose ~= 0
   fprintf( 'Best number of clusters: %d (%.1fs elapsed)\n', ...
         size(estimate.weight,1), toc(tinit) );
end

%disp(['Cfinal = ' num2str(length(inds))]);

% -----------------------------------------------------------

function h = my_plot_init;
   h = figure;

function my_plot_ellipses( h, logF, data, mu, sigma, weight ,u );
   C = size(weight, 2);
   if C==0; return; end

   dtime = 0.3;

   if size(mu,1)<2;
      error('Can not plot 1D objects.');
   end
   %D = size(mu, 1);
   %if D ~= 2
   %  error('Can plot only 2D objects.');
   %end
   dim1 = 1;
   dim2 = 2;

   [x,ids] = sort(weight);

   [x,y,z] = cylinder([2 2], 40);
   xy = [ x(1,:) ; y(1,:) ];

   figure(h);
   clf;

   for i = 1:min(6,C);
     c = ids(i);

     Ldata = logF( data, mu(:,c) );

     subplot( 2, 3, i );
     title(sprintf('Cluster %d\n', c));
     hold on;
     grid on;

     %plot( data(:,dim1,c), data(:,dim2,c), 'rx' );
     %scatter( data(:,dim1,c), data(:,dim2,c), 10, u(:,c) );
     scatter3( Ldata(:,dim1), Ldata(:,dim2), u(:,c), 10, u(:,c) );

     dims   = [dim1,dim2];
     mxy    = chol(sigma(dims,dims,c))' * xy;
     x      = mxy(1,:);
     y      = mxy(2,:);
     z      = ones(size(x))*weight(c);
     plot3( x, y, z, 'k-');
   end
   drawnow;
   hold off


function check_values( values )
   if any(isnan(values(:)))
     error( 'NaNs' );
   elseif any(isinf(values(:)))
     error( 'INF' );
   elseif any(imag(values(:)))
     error( 'Imaginary' );
   end


function [ loglike, L ] = compute_L( indic, alpha, Nparc2, Cnz, N )
   loglike  = sum(log(realmin+sum(indic, 2))); % log P(Y|theta)
   %L      = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*log(N) - loglike;
   L      = Nparc2*sum(log(alpha)) + ...
           Cnz*(Nparc2+0.5)*(1+log(N/12)) - ...
           loglike;


function [ u, indic ] = update_u( logF, N, data, mu, sigma, alpha )
   C = size(sigma,3);
   u = zeros(N,C);   % semi_indic.'
   for c = 1:C
      Ldata    = logF( data, mu(:,c) );
      u(:,c)   = cmvnpdf_clipped( mu(:,c), Ldata, sigma(:,:,c) );
   end
   %check_values(u)
   indic = u .* repmat(alpha, N, 1);


function [ mu, sigma, u, log_fixcount ] = update_cluster( logF, expF, data, D, normindic, mu, conf )
   % Here we need to do a weighted geodesic mean
   mu    = intrinsicMeanWeighted( data, D, logF, expF, normindic, conf, mu );
   %check_values( mu(:,c) );

   % Recalculate Ldata : needs to recalculated if we approximate for the mean
   Ldata = logF( data, mu );

   % Calculate based on covariance type
   normf = 1/sum(normindic);
   aux   = repmat(normindic, 1, D) .* Ldata;

   % Calculate sygma based on the different methods available
   if conf.covtype == 0
     nsigma   = normf*(aux' * Ldata);
   elseif conf.covtype == 1
     nsigma   = normf*diag(sum(aux .* Ldata, 1));
   else
     nsigma   = normf*(aux' * Ldata) .* conf.covtype;
   end
   [sigma log_fixcount] = gmmb_covfixer(nsigma);
   %if log_fixcount > 0;
   %   save( 'bad_matrix.mat', 'normf', 'aux', 'Ldata' );
   %   error( 'bad matrix' );
   %end
   %if log_fixcount > 0;
   %   check_values( sigma );
   %   fprintf( 'Determinant: %.3e, fixed to %.3e with cond %.3e (rank %d)\n', ...
   %          det(nsigma), det(sigma), cond(sigma), rank(sigma) );
   %end
   % covfixer may change the matrix so that log-likelihood
   % decreases. So, if covfixer changes something,
   % disable the stop condition. If going into infinite
   % fix/estimate -loop, quit.

   try
     % Evaluating the belonging of each point to the cluster
     u = cmvnpdf_clipped( mu, Ldata, sigma );
   catch
     error('covariance went bzrk !!!');
   end


% This is known as the standard Expectation Maximization algorithm (EM)
% Classic approach to update everything in block, maybe be slower to converge
function [ alpha, mu, sigma, u, log_fixcount, annihilated_count, Cids ] = ...
     update_block( logF, expF, data, D, alpha, mu, sigma, u, Nparc2, Cids, conf );
   % Get sizes
   [N,C] = size(u);

   % Calculate W weights
   % Must recalculate indic because the alphas can change between generations
   indic       = u .* repmat(alpha, N,1);
   normindic   = indic ./ (realmin + repmat(sum(indic,2), 1,C));

   for c = 1:C;
     % Update the M-step to be able to decide to discard already
     alpha(c) = max(0, sum(normindic(:,c))-Nparc2) / N;
   end
   alpha   = alpha / sum(alpha);
   %check_values(alpha);

   % purge alpha == 0 if necessary
   annihilated_count = length(find(alpha==0));
   if annihilated_count > 0
     nz     = find(alpha>0);
     alpha  = alpha(nz);
     mu     = mu(:,nz);
     sigma  = sigma(:,:,nz);
     u      = u(:,nz);
     C      = length(nz);
     Cids   = Cids(nz);
   end

   % Update the clusters one by one
   log_fixcount = 0;
   for c = 1:C;
     % Update the cluster
     [ nmu, nsigma, nu, fixed ] = update_cluster( logF, expF, data, D, ...
         normindic(:,c), mu(:,c), conf );
     mu(:,c)      = nmu;
     sigma(:,:,c) = nsigma;
     u(:,c)       = nu;
     log_fixcount = log_fixcount + fixed;
   end % while c <= C


% This is known as the Component-Wise EM Algorithm (CEM^2)
% Updates sequentially as described in the paper to avoid no cluster having enough
% important and having them all degrade
function [ alpha, mu, sigma, u, log_fixcount, annihilated_count, Cids ] = ...
     update_sequential( logF, expF, data, D, alpha, mu, sigma, u, Nparc2, Cids, conf );
   % Get sizes
   [N,C] = size(u);

   % Go around randomly to avoid convergence issues
   permid = randperm( C );

   % Update the clusters one by one
   log_fixcount = 0;
   for c = permid;
      % Calculate W weights
      % Must recalculate indic because the alphas can change between generations
      indic       = u .* repmat(alpha, N, 1);
      normindic   = indic ./ (realmin + repmat(sum(indic,2), 1, C));
 
      % Update the M-step to be able to decide to discard already
      alpha(c) = max(0, sum(normindic(:,c))-Nparc2) / N;
      alpha    = alpha / sum(alpha);
      %check_values(alpha);

      % Prune them ahead of time if they must be destroyed
      if alpha(c) <= 0;
         if conf.verbose > 1;
            fprintf( 'Cluster %d / %d annihilated\n', c, C );
         end
         continue; % Ignore this iteration, now it's useless and will get pruned later
      end

      % Update the cluster
      if conf.verbose > 1;
         fprintf( 'Updating cluster %d / %d\n', c, C );
      end
      [ nmu, nsigma, nu, fixed ] = update_cluster( logF, expF, data, D, ...
          normindic(:,c), mu(:,c), conf );
      mu(:,c)        = nmu;
      sigma(:,:,c)   = nsigma;
      u(:,c)         = nu;
      log_fixcount   = log_fixcount + fixed;
   end % while c <= C

   % purge alpha == 0 if necessary
   annihilated_count = length(find(alpha==0));
   if annihilated_count > 0
      nz       = find(alpha>0);
      alpha    = alpha(nz);
      mu       = mu(:,nz);
      sigma    = sigma(:,:,nz);
      u        = u(:,nz);
      Cids     = Cids(nz);
   end



function m = intrinsicMeanWeighted( xi, D, sLog, sExp, weights, conf, varargin )
   % Compute the intrinsic (Frechet) mean of xi
   %
   % At this point we assure nothing for non-localized data.
   % Returns empty matrix on error.

   step     = 1.0; % we may need to change it during iteration
   maxIter  = 1000;

   Ni = size(xi,1); % number of points
   if size(varargin,2) >= 1
      m = varargin{1}; % initial guess
   else
      m = xi(1,:)'; % initial guess
   end
   tol = conf.meantol;

   % Force sparsity by removing insignificant parts (low weights)
   if conf.meanwthresh < 1;
      sumw        = sum(weights);
      [wsort,wids] = sort(weights, 1, 'descend');
      waccum      = 0;
      for j=1:size(wsort,1);
         waccum = waccum + wsort(j);
         if waccum > sumw*conf.meanwthresh; break; end
      end
      wids     = wids(1:j);
      xi       = xi(wids,:);
      weights  = weights(wids);
   end
   N        = size(xi,1);

   % Now the actual process
   normf = 1/sum(weights);
   ww    = repmat(weights,1,D);
   tinit = tic;
   i     = 1;
   while i <= maxIter
      tloop = tic;
      pm    = m;
      
      % iterate
      ss = sLog( xi, m );
      s  = sum(ss.*ww,1);
      m  = sExp( normf*step*s, m )';

      err = norm(pm - m);
      if conf.verbose > 1;
         fprintf('\rmean comp err (iter %d): %.3e (%.1fs) [%d / %d]    ', i, err, toc(tloop), N, Ni );
      end
      if err < tol
         break;
      end
      i = i + 1;
      
      if i == 300 % primitive, change to real line search
         step = 0.5;
      end
   end
   if conf.verbose > 1;
      fprintf(' (total %.1fs)\n', toc(tinit));
   end

   %assert(i <= maxIter);


% Computes the value of a Gaussian PDF
function y = cmvnpdf_clipped( mu, X, Sigma )
   % Get size of data.
   [n,d]       = size(X);
   invSigma    = inv(Sigma);
   sqrdist     = sum((X*invSigma).*conj(X),2);
   invDetSigma = 1/real(det(Sigma));

   y = sqrt( (2*pi)^(-d) * invDetSigma ) .* exp(-0.5*real(sqrdist));




