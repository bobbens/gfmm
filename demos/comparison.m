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

C  = [1 1 1];

tol = 1e-5;

% setup manifold: quadratic surface
F     = @(x) sum(C'.*(x.^2))-1.0;
DF    = @(x) 2*C.*x';
D2F   = @(x) 2.0*diag(C);
manifold = embeddedManifold( m, n, F, DF, D2F, tol );

% create Gaussians
NEACH    = 1000;
COVVAR   = [ 0.2, 0; 0, 0.3 ];
colors   = { 'b', 'g', 'r', 'c', 'm', 'y' };
Gdata = struct;
for i=1:6;
   Gdata(i).mean  = [cos(i*(pi/3));0;sin(i*(pi/3))];
   Gdata(i).col   = colors{i};
   Gdata(i).cov   = COVVAR;
   Gdata(i).n     = NEACH;
end

results = zeros( 100, 3 );
for r=1:size(results,1);
   for g=1:size(Gdata,2);
      gcov  = Gdata(g).cov;
      gn    = Gdata(g).n;
      data  = randn(gn,2)*gcov;
      Gdata(g).data = data';
   end

   % Get functions
   [ logF, expF ] = sphereFunc();

   % Draw the gaussians
   Xdata = [];
   for g=1:size(Gdata,2);
      p     = manifold.toManifold( Gdata(g).mean );

      Edata = expF( Gdata(g).data', p );

      % Plot data
      for i = 1:Gdata(g).n;
         x     = Edata(i,:)';
         Xdata = [Xdata,x];
      end
   end

   mu    = umean( Xdata', 3, logF, expF ); 
   Ldata = logF(  Xdata', mu );

   % Perform GGMM
   [ estimate ] = gfmm( Xdata', 2, logF, expF, ...
         'verbose',  0, ...
         'animate',  0, ...
         'thr',      1e-6, ...
         'maxloops', 500, ...
         'Cmax',     50, ...
         'covtype',  0, ... % 0 is full, 1 is diagonal, mask is whatever that is
         'logging',  0 );
         %'covinit',  3, ...
   results( r, 1 ) = size(estimate.mu,2);

   % Vanilla PGA
   [ estimate ] = gmm_vanilla( Ldata, ...
         'verbose',  0, ...
         'animate',  0, ...
         'Cmax',     30, ...
         'covtype',  0, ... % 0 is full, 1 is diagonal, mask is whatever that is
         'logging',  0 );
   results( r, 2 ) = size(estimate.mu,2);

   % Von Mises-Fisher
   [ estimate ] = gmm_movmf( Xdata', ...
         'verbose',  0, ...
         'animate',  0, ...
         'Cmax',     30, ...
         'covtype',  0, ... % 0 is full, 1 is diagonal, mask is whatever that is
         'logging',  0 );
   results( r, 3 ) = size(estimate.mu,2);

   fprintf( '%03d) GFMM=%d, PGA=%d, movMF=%d\n', r, results(r,1), results(r,2), results(r,3) );
end


