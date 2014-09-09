function Exp = ExpNP(v)
% EXPNP Riemannian exponential map at North pole of S^2
%       ExpNP(v) returns 3 x n matrix where each column is a point on a 
%                sphere and the input v is 2 x n matrix where each column 
%                is a point on tangent  space at north pole.
%
%
%   See also LogNP.

% Last updated Aug 10, 2009
% Sungkyu Jung

v1    = v(1,:);
v2    = v(2,:);
normv = sqrt(v1.^2+v2.^2);
Exp   = real( [v1.*sin(normv)./normv; v2.*sin(normv)./normv; cos(normv)] );
Exp(:,normv < 1e-16) = repmat([0 ;0 ;1],1,sum(normv < 1e-16));
