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
#ifndef __itkMultigridWeightedJacobiSmoother_h
#define __itkMultigridWeightedJacobiSmoother_h

#include "itkMultigridSmoother.h"
#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

/** \class MultigridWeightedJacobiSmoother
 *
 * \brief Weighted Jacobi smoother implementation; it is a derived
 * class of MultigridSmoother, and its constructor can optionally
 * take the weight coefficient as an argument, which otherwise defaults
 * to \f$ \frac{2}{3} \f$.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class MultigridWeightedJacobiSmoother : public MultigridSmoother< VDimension >
{
public:

  /** Standard class typedefs. */
  typedef double                                                    Precision;
  typedef MultigridWeightedJacobiSmoother                           Self;

  typedef Image< Precision, VDimension >                            ImageType;
  typedef Neighborhood< Precision, VDimension >                     StencilType;
  typedef StencilImage< Precision, VDimension >                     StencilImageType;

  typedef ImageRegion< VDimension >                                 ImageRegionType;
  typedef typename ImageType::SpacingType                           SpacingType;
  typedef typename ImageType::SizeType                              SizeType;
  typedef typename ImageType::IndexType                             IndexType;
  typedef typename ImageType::OffsetType                            OffsetType;
  typedef typename std::list< OffsetType >                          OffsetListType;


  /** Returns the approximated solution to problem \f$ A x = b\f$ after one iteration of the smoother,
   * starting with inputImage as initial guess, rhsImage as the right hand side \f$ b \f$, and
   * matrixImage as matrix \f$ A \f$ in StencilImageType format. */
  typename ImageType::Pointer SingleIteration( const ImageType * inputImage,
                                               const ImageType * rhsImage,
                                               const StencilImageType * matrixImage ) const;

  /** Returns the residual \f$ r = b - A x \f$, where inputImage is the term \f$ x \f$, rhsImage is
   * \f$ b \f$, and matrixImage the matrix \f$ A \f$. */
  typename ImageType::Pointer ComputeResidual( const ImageType * inputImage,
                                               const ImageType * rhsImage,
                                               const StencilImageType * matrixImage ) const;

  /** Class constructor. */
  MultigridWeightedJacobiSmoother( Precision weight );

  /** Class contructor, with default weight \f$ \frac{2}{3} \f$. */
  MultigridWeightedJacobiSmoother();

  /** Class destructor. */
  ~MultigridWeightedJacobiSmoother();


private:

  Precision                  m_Omega;


  /** Purposely not implemented */
  MultigridWeightedJacobiSmoother( const Self & );
  Self & operator= ( const Self & );

};


} // end namespace mad

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultigridWeightedJacobiSmoother.hxx"
#endif

#endif /* __MultigridWeightedJacobiSmoother_h */
