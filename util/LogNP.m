function Log = LogNP(vdata)
% LOGNP Riemannian log map at North pole of S^2
%       LogNP(vdata) returns 2 x n matrix where each column is a point on tangent
%       space at north pole and the input vdata is 3 x n matrix where each column 
%       is a point on a sphere.
%
%
%   See also ExpNP.

% Last updated Aug 10, 2009
% Sungkyu Jung
scale = acos(vdata(3,:)) ./ sqrt(1-vdata(3,:).^2);
scale(isnan(scale)) = 1;
Log = [vdata(1,:).*scale ; vdata(2,:).*scale ];


