function [ logfunc, expfunc ] = surfaceFunc( manifold )

   logfunc = @s_logfunc;
   expfunc = @s_expfunc;

   function u = s_logfunc( data, mu )
      B = manifold.orthFrame(mu);
      N = size(data,1); % number of points
      u = zeros(manifold.dim,N);
      parfor j = 1:N;
         u(:,j) = B'*manifold.Log( mu, data(j,:)' );
      end
      u = u';
   end

   function x = s_expfunc( data, mu )
      B = manifold.orthFrame(mu);
      N = size(data,1);
      parfor j = 1:N;
         x(:,j) = manifold.Exp( mu, B*data(j,:)' );
      end
      x = x';
   end
end

