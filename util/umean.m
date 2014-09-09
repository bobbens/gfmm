function m = umean( x, D, sLog, sExp )
   % Compute the intrinsic (Frechet) mean of xi
   %
   % At this point we assure nothing for non-localized data.
   % Returns empty matrix on error.

   step     = 1.0; % we may need to change it during iteration
   maxIter  = 1000;

   N     = size(x,1); % number of points
   m     = x(1,:)'; % initial guess
   tol   = 1e-10;

   % Now the actual process
   tinit = tic;
   i     = 1;
   while i <= maxIter
      tloop = tic;
      pm    = m;
      
      % iterate
      ss = sLog( x, m );
      s  = mean(ss,1);
      m  = sExp( step*s, m )';

      err = norm(pm - m);
      %fprintf('\rmean comp err (iter %d): %.3e (%.1fs)  ', i, err, toc(tloop));
      if err < tol
         break;
      end
      i = i + 1;
      
      if i == 300 % primitive, change to real line search
         step = 0.5;
      end
   end
   %fprintf(' (total %.1fs)\n', toc(tinit));

   %assert(i <= maxIter);

