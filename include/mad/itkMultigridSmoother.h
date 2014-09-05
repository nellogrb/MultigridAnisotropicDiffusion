/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkMultigridSmoother_h
#define __itkMultigridSmoother_h

#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

/** \class MultigridSmoother
 *
 * \brief Base class for a generic smoother. The derived class has to provide
 * an implementation of the methods SingleIteration and ComputeResidual defined
 * in the public section, which refer to the problem
 *
 * \f[
 *   A x = b
 * \f]
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class ITK_EXPORT MultigridSmoother
{
public:

  /** Standard class typedefs. */
  typedef double                                                    Precision;
  typedef MultigridSmoother                                         Self;

  typedef Image< Precision, VDimension >                            ImageType;
  typedef StencilImage< Precision, VDimension >                     StencilImageType;


  /** Returns the approximated solution to problem \f$ A x = b\f$ after one iteration of the smoother,
   * starting with inputImage as initial guess, rhsImage as the right hand side \f$ b \f$, and
   * matrixImage as matrix \f$ A \f$ in StencilImageType format. */
  virtual typename ImageType::Pointer SingleIteration( const ImageType * inputImage,
                                                       const ImageType * rhsImage,
                                                       const StencilImageType * matrixImage ) const = 0;

  /** Returns the residual \f$ r = b - A x \f$, where inputImage is the term \f$ x \f$, rhsImage is
   * \f$ b \f$, and matrixImage the matrix \f$ A \f$. */
  virtual typename ImageType::Pointer ComputeResidual( const ImageType * inputImage,
                                                       const ImageType * rhsImage,
                                                       const StencilImageType * matrixImage ) const = 0;
  /** Class constructor */
  MultigridSmoother();

  /** Class destructor. */
  virtual ~MultigridSmoother();

};


template < unsigned int VDimension >
MultigridSmoother< VDimension >
::MultigridSmoother() {}

template < unsigned int VDimension >
MultigridSmoother< VDimension >
::~MultigridSmoother() {}


} // end namespace mad

} // end namespace itk

#endif /* __itkMultigridSmoother_h */
