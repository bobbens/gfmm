GFMM
====


OVERVIEW
--------

This code provides an implementation of the research paper:

```
  Geodesic Finite Mixture Models
  E. Simo-Serra, C. Torras, and F. Moreno-Noguer
  British Machine Vision Conference (BMVC), 2014
```

This allows clustering of data that is located on a known Riemannian manifold. Some highlights of the algorithm:

* Generative model
* Fully unsupervised
* Scales to large data


License
-------

```
  Copyright (C) <2014> <Edgar Simo-Serra>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the version 3 of the GNU General Public License
  as published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.      
  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  Edgar Simo-Serra, Institut de Robotica i Informatica Industrial (CSIC/UPC), September 2014.
  esimo@iri.upc.edu, http://www-iri.upc.es/people/esimo/
```


Installation
------------

The visualization code is dependent on Stefan Sommer's "smanifold" code. This can be found at https://github.com/nefan/smanifold . Once it is installed, edit the path in startup.m to point to the path in which this code is installed. As the rest of the code is pure matlab, there is no need to compile nor install anything else.


Usage
-----

There are two main files for the approach:

* gfmm.m -- Performs the model estimation
* gfmm_sample.m -- Performs sampling on the model

Additionally some demos are provided in demos/:

* sphere.m -- Synthetic sphere example.
* quad_surface.m -- Quadratic surface example. Very slow to compute as there are no closed forms for the exponential and logarithmic map operators.
* comparison.m -- Compares our model (GFMM) with von Mises-Fischer distributions and using a single tangent plane.

These demos reproduce several of the results in the paper. They were all run with "MATLAB R2012a (7.14.0.739) 64-bit (glnxa64)" and used a fixed seed for the random number generator to be deterministic. On different set-ups they may give different results.

In order to see results fast it is recommended to try the sphere demo as following:

```
>> sphere
```

You should get some output as follows:

```
 Initial distributions set up in 0.0 seconds
 Cnz=30, t=1 => 3.755e+02 <? 4.159e-03  (L=4.068e+03) in 0.4 secs
 Cnz=30, t=2 => 4.611e+01 <? 3.784e-03  (L=4.021e+03) in 0.8 secs
 Cnz=30, t=3 => 2.287e+01 <? 3.738e-03  (L=3.997e+03) in 0.5 secs
 Cnz=30, t=4 => 2.827e+01 <? 3.715e-03  (L=3.968e+03) in 0.5 secs
 Cnz=30, t=5 => 5.060e+01 <? 3.687e-03  (L=3.917e+03) in 0.6 secs
 Cnz=30, t=6 => 9.567e+01 <? 3.636e-03  (L=3.820e+03) in 0.5 secs
 Cnz=30, t=7 => 1.207e+02 <? 3.540e-03  (L=3.697e+03) in 0.5 secs
 ...
```

Additionally some figures will show results and progress.

If you use this code please cite:

```
 @InProceedings{SimoSerraBMVC2014,
    author = {Edgar Simo-Serra and Carme Torras and Francesc Moreno-Noguer},
    title = {{Geodesic Finite Mixture Models}},
    booktitle = "Proceedings of the British Machine Vision Conference (BMVC)",
    year = 2014,
 }
```


Changelog
---------

September 2014: Initial version 1.0 release


