.. _matrix_aggregating_methods:

Matrix aggregating methods
==========================

SpAbs
-----
.. math::
    {\rm SpAbs} = \sum_{i = 1}^N \left| \lambda_i \right|

where :math:`\lambda_i` is :math:`i`-th eigenvalue.

SpMax
-----
.. math::
    {\rm SpMax} = \max_{i = 1}^N \lambda_i

SpDiam
------
.. math::
    {\rm SpDiam} = {\rm SpMax} - {\rm SpMin}

SpAD
----
.. math::
    {\rm SpAD} = \sum_{i = 1}^N \left| \lambda_i - \bar{\lambda} \right|

SpMAD
-----
.. math::
    {\rm SpMAD} = \frac{\rm SpAD}{A}

where :math:`A` is number of atoms.

LogEE
-----
.. warning::

   Despite its name, this descriptor does **not** calculate the published
   :math:`\mathrm{LogEE}` value. Due to an implementation issue in the original
   implementation, it instead computes

   .. math::

      \log\!\left(1 + \sum_{i=1}^{N} \exp(\lambda_i)\right).

   This behavior has **not** been corrected in order to preserve backwards
   compatibility with previous versions and existing descriptor values.

   The equation below is the definition given in the original documentation and
   is retained here solely for historical completeness. It does **not** describe
   the value returned by the current implementation.

**Published definition (historical):**

.. math::
    {\rm LogEE} = \log\!\left(\sum_{i=1}^{N} \exp(\lambda_i)\right)

SM1
---
.. math::
    {\rm SM1} = \sum_{i = 1}^N \lambda_i

VE1
---
.. math::
    {\rm VE1} = \sum_{i = 1}^N \left| \ell_i \right|

where :math:`\ell_i` is eigenvector elements corresponding to leading eigenvalue.

VE2
---
.. math::
    {\rm VE2} = \frac{\rm VE1}{A}

VE3
---
.. math::
    {\rm VE3} = \log(\frac{A}{10} \cdot {\rm VE1})

NaN when :math:`{\rm VE1} = 0`.

VR1
---
.. math::
    {\rm VR1} = \sum_{(i, j) \in {\rm bonds}} \left( \ell_i \cdot \ell_j \right)^{-1/2}

VR2
---
.. math::
    {\rm VR2} = \frac{\rm VR1}{A}

VR3
---
.. math::
    {\rm VR3} = \log(\frac{A}{10} \cdot {\rm VR1})

NaN when :math:`{\rm VR1} = 0`.
