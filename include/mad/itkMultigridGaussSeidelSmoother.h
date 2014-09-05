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
#ifndef __itkMultigridGaussSeidelSmoother_h
#define __itkMultigridGaussSeidelSmoother_h

#include "itkMultigridSmoother.h"
#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

/** \class MultigridGaussSeidelSmoother
 *
 * \brief Gauss Seidel smoother implementation; it is a derived
 * class of MultigridSmoother. It internally uses a lexicographic
 * ordering of the points.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class ITK_EXPORT MultigridGaussSeidelSmoother : public MultigridSmoother< VDimension >
{
public:

  /** Standard class typedefs. */
  typedef double                                                    Precision;
  typedef MultigridGaussSeidelSmoother                              Self;

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

  /** Class constructor */
  MultigridGaussSeidelSmoother();

  /** Class destructor. */
  ~MultigridGaussSeidelSmoother();


private:

  /** Utility function which returns whether, in lexicographic ordering, offset
   *  left comes before or after offset right. */
  inline bool LexOrder( const OffsetType left, const OffsetType right ) const
  {

    for ( int i = VDimension - 1; i >= 0; --i )
      {

      if ( left[ i ] < right[ i ] ) return true;
      else if ( left[ i ] > right[ i ] ) return false;

      }

    return false;

  }


  /** Purposely not implemented */
  MultigridGaussSeidelSmoother( const Self & );
  Self & operator= ( const Self & );

};


} // end namespace mad

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultigridGaussSeidelSmoother.hxx"
#endif

#endif /* __itkMultigridGaussSeidelSmoother_h */
