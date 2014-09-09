function [estimate, varargout] = gmm_movmf(data, varargin);
if any( abs( sqrt( sum( data .^2, 2 ) ) -1 ) > 1e-10 );
   error( 'Data is not on a hypersphere.' );
end

[N, D] = size(data);	% number of points (n), dimensions (d)

% defaults
conf = struct(...
	'maxloops', 500, ...
	'Cmax', ceil(min(50, N/(D*D)/3)), ...
	'Cmin', 1, ...
	'verbose', 0, ...
	'thr', 1e-6, ...
	'animate', 0, ...
	'covtype', 0, ...
   'covinit', 0, ...
	'broken', 1, ...
	'logging', 0, ...
	'logfile', 'logging.mat' ...
	);

if nargout>1
	conf.logging = 1;
	varargout{1} = [];
end

conf = getargs(conf, varargin);

C = conf.Cmax;

if nargout<2
	conf.logging=0;
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


if (C<1) | (C>N)
	C  = N;
	mu = data.';
else
	% initialize mu as random points from data
	permi = randperm(N);
	mu    = data(permi(1:C),:).';  % D x C
end


% initialize kappa
if conf.covinit==0;
   s2    = 20*pi / sqrt(C);
else
   s2    = 20*pi / conf.covinit;
end
kappa = ones(1,1,C)*10;

% weights initialization
alpha = ones(1,C) * (1/C);


log_initialmix = struct(...
	'weight', alpha, ...
	'mu', mu, ...
	'kappa', kappa);


% the number of free parameters in a Gaussian
Nparc    = D-1+1; % Kappa is always 1D and vectors so it's smaller
Nparc2   = Nparc/2;

N_limit = (Nparc+1)*3*conf.Cmin;
if N < N_limit
	warning_wrap('gmmb_fj:data_amount', ...
	   ['Training data may be insufficient for selected ' ...
	    'minimum number of components. ' ...
	    'Have: ' num2str(N) ', recommended: >' num2str(N_limit) ...
	    ' points.']);
end


if conf.animate ~= 0
	aniH = my_plot_init;
	my_plot_ellipses(aniH, data, mu, kappa, alpha);
end


t     = 0;
Cnz   = C;	% (k_nz) k = kmax
Lmin  = NaN;
Cids  = 1:C;
tbest = -1;
tinit = tic();

u = zeros(N,C);	% semi_indic.'
for c = 1:C
	u(:,c) = vmf_pdf(data, mu(:,c).', kappa(:,:,c), D);
end
indic = u .* repmat(alpha, N,1);

old_loglike = sum(log(sum(realmin+indic, 2)));
%old_L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*log(N) - old_loglike;
old_L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*(1+log(N/12)) - old_loglike;

while Cnz >= conf.Cmin
	repeating      = 1;
	
	fixing_cycles  = 0;
	loops          = 0;
	while repeating
		t = t+1;
		loops = loops +1;
		
		fixed_on_this_round  = 0;
		log_covfixer2{t,1}   = 0;
      permid               = randperm( C ); % Go around randomly
      cid                  = 0;
      for c = permid;
         if conf.verbose;
            cid = cid+1;
            fprintf( '\r%03d) Cluster %05d / %05d    ', t, cid, C );
         end

			indic       = u .* repmat(alpha, N,1);
			normindic   = indic ./ (realmin + repmat(sum(indic,2), 1, C));

			normf       = 1/sum(normindic(:,c));
         if isinf(normf)
            %warning( 'normf went to inf' );
            alpha(c) = 0;
         else

            aux         = repmat(normindic(:,c), 1, D) .* data;
            
            nmu         = normf * sum(aux,1);
            mu(:,c)     = (nmu.') ./ norm(nmu.');
            assert( all(isfinite(mu(:))) )

            if 0;
            figure(c);
            clf;
            hold on;
            plot3(data(:,1),data(:,2),data(:,3),'ro')
            aa = N*normf*aux;
            plot3(aa(:,1),aa(:,2),aa(:,3),'go')
            plot3(mu(1,c),mu(2,c),mu(3,c),'bx', 'MarkerSize', 30);
            oaeu
            end

            kappa(:,:,c)  = vmf_kappa( normf*aux );
            %aoeu
            
            alpha(c) = max(0, sum(normindic(:,c))-Nparc2) / N;
            if ~isfinite(alpha(c))
               alpha(c) = 0;
            end
         end
		
			if ~all( isfinite(alpha(:)) )
				% something went wrong
				% probably there is not enough data to
				%support estimation
            alpha
				warning('gmmb_fj:weight_finity', 'Mixture weights are no longer finite, aborting estimation.');
				alpha(:) = 0;
				Cnz = 0;
				repeating = 0;
            estimate          = struct;
            estimate.mu       = [];
            estimate.kappa    = [];
            estimate.weight   = [];
            return
			end
			alpha    = alpha / sum(alpha);

			if alpha(c) == 0
				Cnz = Cnz - 1;
			else
				%try
					u(:,c) = vmf_pdf( data, mu(:,c).', kappa(:,:,c), D );
				%catch
			  	%	disp('covariance went bzrk !!!');
				%	kappa(:,:,c)
					%keyboard
				%	Cnz = 0;
				%end
			end
			c=c+1;
			
			if Cnz <= 0
				% number of components fell to zero
				% nothing can be done
				error('Estimation failed, number of components fell to zero. Not enough training data?');
			end

		end % while c <= C

		% purge alpha == 0 if necessary
		annihilated_count = length(find(alpha==0));
		if annihilated_count > 0
			nz    = find(alpha>0);
			alpha = alpha(nz);
			mu    = mu(:,nz);
			kappa = kappa(:,:,nz);
			u     = u(:,nz);
			C     = length(nz);
		end

		if conf.animate ~= 0
			my_plot_ellipses(aniH, data, mu, kappa, alpha);
		end
		
		u = zeros(N,C);	% semi_indic.'
		for c = 1:C
			u(:,c) = vmf_pdf(data, mu(:,c).', kappa(:,:,c), D);
		end
		indic = u .* repmat(alpha, N,1);
		
		loglike = sum(log(realmin+sum(indic, 2)));
		%L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*log(N) - loglike;
      L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*(1+log(N/12)) - loglike;

		
		if conf.verbose ~= 0
         fprintf( '%.3f <? %.3f  (L=%.3f)\n', ...
               abs(loglike - old_loglike), ...
               conf.thr*abs(old_loglike), L );
			%disp(['Cnz=' num2str(Cnz) ' t=' num2str(t) ' '...
			%   num2str(abs(loglike - old_loglike)) ...
			%   ' <? ' num2str(conf.thr*abs(old_loglike))]);
			%disp(['t=' num2str(t) ' L= ' num2str(L)]);
		end
		
		if conf.logging>0
			log_loglikes{t} = loglike;
			log_costs{t} = L;
			log_annih{t} = [annihilated_count, 0];
		end
		if conf.logging>1
			log_mixtures{t} = struct(...
				'weight', alpha, ...
				'mu', mu, ...
				'kappa', kappa);
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
			if (abs(loglike/old_loglike-1) < conf.thr)
				repeating = 0;
			end
		end
		
		old_L = L;
		old_loglike = loglike;
		
		if fixing_cycles > 20
			repeating = 0;
		end
		if loops > conf.maxloops
			repeating = 0;
		end
	end % while repeating
	
	if isnan(Lmin) | (L <= Lmin)
      tbest = t;
		Lmin  = L;
		estimate = struct('mu', mu,...
			'kappa', kappa,...
			'weight', alpha.');
	end
	if conf.verbose ~= 0
		disp(['Cnz = ' num2str(Cnz)]);
	end

	% annihilate the least probable component
	m = find(alpha == min(alpha(alpha>0)));
	alpha(m(1)) = 0;
	Cnz = Cnz -1;
	% alpha doesn't need to be normalized here, even if it would seem logical to do so.
	
	if conf.logging > 0
		log_annih{t}(2) = 1;
	end
	
	if Cnz > 0
		alpha = alpha / sum(alpha);
	
		% purge alpha == 0 if necessary
		if length(find(alpha==0)) > 0
			nz = find(alpha>0);
			alpha = alpha(nz);
			mu = mu(:,nz);
			kappa = kappa(:,:,nz);
			u = u(:,nz);
			C = length(nz);
		end
		
		u = zeros(N,C);	% semi_indic.'
		for c = 1:C
			u(:,c) = vmf_pdf(data, mu(:,c).', kappa(:,:,c), D);
		end
		indic = u .* repmat(alpha, N,1);
		
		old_loglike = sum(log(realmin+sum(indic, 2)));
		%old_L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*log(N) - old_loglike;
		old_L = Nparc2*sum(log(alpha)) + (Nparc2+0.5)*Cnz*(1+log(N/12)) - old_loglike;
	end
end

if conf.logging>1
	varargout{1} = struct(...
		'iterations', {t}, ...
		'costs', {cat(1,log_costs{:})}, ...
		'annihilations', {sparse(cat(1,log_annih{:}))}, ...
		'covfixer2', {cat(1,log_covfixer2{:})}, ...
		'loglikes', {cat(1,log_loglikes{:})}, ...
		'initialmix', {log_initialmix}, ...
		'mixtures', {log_mixtures});
end
if conf.logging == 1
	varargout{1} = struct(...
		'iterations', {t}, ...
		'costs', {cat(1,log_costs{:})}, ...
		'annihilations', {sparse(cat(1,log_annih{:}))}, ...
		'covfixer2', {cat(1,log_covfixer2{:})}, ...
		'loglikes', {cat(1,log_loglikes{:})} ...
		);
end

% purge alpha==0
e = estimate;
inds = find(e.weight>0);
estimate.mu = e.mu(:,inds);
estimate.kappa = e.kappa(:,:,inds);
estimate.weight = e.weight(inds);

if conf.animate ~= 0
	my_plot_ellipses(aniH, data, estimate.mu, estimate.kappa, estimate.weight);
end

%disp(['Cfinal = ' num2str(length(inds))]);
end

% -----------------------------------------------------------

function h = my_plot_init;
   h = figure;
   figure(h);
   title('Distribution of x_1 and x_2 values','FontSize',14);
   xlabel('x_1 value','FontSize',14);
   ylabel('x_2 value','FontSize',14);
   zlabel('weight','FontSize',14);
   view(2)
   tic;
end

function my_plot_ellipses(h, data, mu, kappa, weight);
   warning( 'unimplemented for vmf' )
   return
   dtime = 0.3;

   %D = size(mu, 1);
   %if D ~= 2
   %	error('Can plot only 2D objects.');
   %end

   [x,y,z] = cylinder([2 2], 40);
   xy = [ x(1,:) ; y(1,:) ];

   figure(h);

   plot(data(:,1), data(:,2), 'rx');

   hold on
   C = size(mu, 2);
   for c = 1:C
      mxy = chol(kappa(1:2,1:2,c))' * xy;
      x = mxy(1,:) + mu(1,c);
      y = mxy(2,:) + mu(2,c);
      z = ones(size(x))*weight(c);
      plot3(x,y,z, 'k-');
   end
   drawnow;
   hold off

   t = toc;
   if t+0.01<dtime
      pause(dtime-t);
   end
   tic
end

function u = vmf_pdf( data, mu, kappa, D )
   C = (D/2-1)*log(kappa) - D/2*log(2*pi) - log(besseli(D/2-1,kappa));
   E = kappa * (data*mu');
   u = exp( C + E );
   assert( all(isfinite(u)) );
   return

   %if (exp(kappa)-exp(-kappa)) > 1e100;
   %   C = 0;
   %else
   %   C = kappa / (2*pi*(exp(kappa)-exp(-kappa)));
   %end
   %C = kappa / (2*pi*(exp(kappa)-exp(-kappa)))
   bi = besseli(D/2-1,kappa);
   C  = kappa^(D/2-1) / ((2*pi)^(D/2)*bi);
   %assert( isfinite(C) )
   %C  = 2 / kappa * (exp(kappa)-exp(-kappa)/2);
   u  = C .* exp( kappa * (data*mu') );
   assert( all(isfinite(u)) )
end

function kappa = vmf_kappa( data )
   p  = size(data,2);
   R  = norm(sum(data,1));
   %mm = mean(data,1);
   %plot3( [0 mm(1)], [0 mm(2)], [0 mm(3)] );
   R2 = R^2;
   kappa = R*(p-R2)/(1-R2);
   assert(isfinite(kappa))

   for i=1:10;
      Ap    = besseli(p/2,kappa) / besseli(p/2-1,kappa);
      if ~isfinite(Ap); % Shouldn't be happening however...
         break;
      end
      k     = kappa - (Ap-R)/(1-Ap^2-((p-1)/kappa)*Ap);

      % Break if change is small
      if abs(k-kappa)/kappa < 1e-3;
         break;
      end

      kappa = k;
   end
   if kappa < 0;
      mean(data,1)
      p
      R
      R2
      kappa
      error( 'negative kappa' )
   end
   %if R < 0.53
   %  kappa = 2*R + R^3 + 5*R^5/6;
   %elseif R>=0.53 && R<0.85
   %  kappa = -.4 + 1.39*R + 0.43/(1-R);
   %else
   %  kappa = 1/(R^3 - 4*R^2 + 3*R);
   %end
end

