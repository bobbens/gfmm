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


% Same seed to produce deterministic results
seed = 42;
randn(   'state', seed );
rand(    'state', seed );

% Dimensions
m  = 3; % Embedding space
n  = 1;

C  = [1 1 1]; % These are the derivatives

tol = 1e-5;

% setup manifold: quadratic surface
F     = @(x) sum(C'.*(x.^2))-1.0;
DF    = @(x) 2*C.*x';
D2F   = @(x) 2.0*diag(C);
manifold = embeddedManifold( m, n, F, DF, D2F, tol );

% create Gaussians
NEACH    = 300;
COVVAR   = [ 0.2, 0; 0, 0.3 ];
colors   = { 'b', 'g', 'r', 'c', 'm', 'y' };
Gdata = struct;
for i=1:6;
   Gdata(i).mean  = [cos(i*(pi/3));0;sin(i*(pi/3))];
   Gdata(i).col   = colors{i};
   Gdata(i).cov   = COVVAR;
   Gdata(i).n     = NEACH;
end

for g=1:size(Gdata,2);
   gcov  = Gdata(g).cov;
   gn    = Gdata(g).n;
   data  = randn(gn,2)*gcov;
   Gdata(g).data = data';
end

% Get functions
[ logF, expF ] = sphereFunc();

% Draw the gaussians
figure(1);
clf;
xrange = [-2.5,2.5];
yrange = [-2.5,2.5];
zrange = [-2.5,2.5];
ImplicitPlot3D( F, xrange, yrange, zrange, 50 );
hold on;
Xdata = [];
for g=1:size(Gdata,2);
   p     = manifold.toManifold( Gdata(g).mean );

   % Plot center
   plot3( p(1), p(2), p(3), 'ko', ...
         'MarkerSize', 12, 'MarkerFaceColor', Gdata(g).col );

   Edata = expF( Gdata(g).data', p );

   % Plot data
   for i = 1:Gdata(g).n;
      x  = Edata(i,:)';
      plot3( x(1), x(2), x(3), 'ro', ...
           'MarkerSize', 10, ...
           'MarkerEdgeColor', 'k', ...
           'MarkerFaceColor', Gdata(g).col );
      Xdata = [Xdata,[x;g]];
   end
end
hold off;
axis( [-2.5 2.5 -2.5 2.5 -2.5 2.5] );
view( [16 22] );

% Estimate GFMM
[ estimate, logging ] = gfmm( Xdata(1:3,:)', 2, logF, expF, ...
      'verbose',  1, ...
      'animate',  1, ...
      'Cmax',     30, ...
      'covtype',  0, ... % 0 is full, 1 is diagonal, mask is whatever that is
      'logging',  2, ...
      'logfile',  'sphere.mat' );



