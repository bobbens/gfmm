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

c  = -2; % get coefficient from parameters
C  = [c(1) 1 1];

tol = 1e-5;

% setup manifold: quadratic surface
F     = @(x) sum(C'.*(x.^2))-1.0;
DF    = @(x) 2*C.*x';
D2F   = @(x) 2.0*diag(C);
manifold = embeddedManifold( m, n, F, DF, D2F, tol );

% create Gaussians
NEACH = 300;
Gdata = struct;
Gdata(1).mean  = [1;-1;1];
Gdata(1).cov   = [ 0.2, 0; 0, 0.3 ];
Gdata(1).n     = NEACH;
Gdata(1).col   = 'r';
Gdata(2).mean  = [0;0;1];
%Gdata(2).cov   = [ 0.3, 0; 0, 0.1 ];
Gdata(2).cov   = [ 0.25, 0; 0, 0.1 ];
Gdata(2).n     = NEACH;
Gdata(2).col   = 'g';
Gdata(3).mean  = [0.2;-1;1];
Gdata(3).cov   = [ 0.2, 0; 0, 0.1 ];
Gdata(3).n     = NEACH;
Gdata(3).col   = 'b';
Gdata(4).mean  = [0;0;1];
Gdata(4).cov   = [ 0.1, 0; 0, 0.25 ];
Gdata(4).n     = NEACH;
Gdata(4).col   = 'k';
Gdata(5).mean  = [0;-1;0.5];
Gdata(5).cov   = [ 0.25, 0; 0, 0.1 ];
Gdata(5).n     = NEACH;
Gdata(5).col   = 'c';
for g=1:size(Gdata,2);
   gcov  = Gdata(g).cov;
   gn    = Gdata(g).n;
   data  = randn(gn,2)*gcov;
   Gdata(g).data = data';
end

% Get functions
[ logF, expF ] = surfaceFunc( manifold );

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
            'MarkerSize', 5, ...
            'MarkerEdgeColor', 'k', ...Gdata(g).col, ...
            'MarkerFaceColor', Gdata(g).col );
      Xdata = [Xdata,[x;g]];
   end
end
hold off;
axis( [-2.5 2.5 -2.5 2.5 -2.5 2.5] );
view( [16 22] );

% Estimate Mixture
[ estimate, logging ] = gfmm( Xdata(1:3,:)', 2, logF, expF, ...
      'verbose',  2, ...
      'animate',  0, ...
      'Cmax',     10, ...
      'covtype',  0, ...
      'meantol',  1e-6, ...
      'meanwthresh', 0.999, ...
      'thr',      1e-5, ...
      'logging',  2, ...
      'logfile',  'quad_surface.mat' )



