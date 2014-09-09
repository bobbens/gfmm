function [ samples ] = gfmm_sample( mix, n, expF )
%GFMM_SAMPLE Samples from a Geodesic Finite Mixture Model.
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
% SYNOPSIS samples = gfmm_sample( mix, n, expF )
%
% INPUT mix: Geodesic Finite Mixture Model to sample from
%       n: Number of samples to get
%       expF: Exponential map function for the manifold.
%
% OUTPUT samples: Matrix containing all the samples from the mixture projected on the manifold.
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
% SEE ALSO gfmm
%
% Author: Edgar Simo-Serra
% Date: 08-Sep-2014
% Version: 1.0
   [sortw, idsw] = sort( mix.weight, 'descend' );
   cvals    = rand( n, 1 ); % Clusters to sample from
   nc       = size( mix.weight, 2 );
   dM       = size( mix.mu, 1 );
   dT       = size( mix.sigma, 1 );
   samples  = zeros( n, dM );
   waccum   = 0;
   ndone    = 0;
   for i=1:nc;
      % Determine clusters
      ids   = ((cvals < waccum + sortw(i)) & (cvals > waccum));
      nsamp = sum(ids);

      if nsamp > 0;
         % Sample
         Tsamp = randn( nsamp, dT ) * mix.sigma( :, :, idsw(i) );

         % Project to manifold
         Msamp = expF( Tsamp, mix.mu( :, idsw(i) ) );

         % Combine
         samples( ndone+1:ndone+nsamp, : ) = Msamp;

         % Already done
         ndone = ndone + nsamp;
         if ndone>=n; return; end
      end
      
      % Increment weight
      waccum = waccum + sortw(i);
   end



