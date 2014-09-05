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
#ifndef __itkDirectSolver_h
#define __itkDirectSolver_h

#include "vnl/vnl_sparse_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_sparse_lu.h"

#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

/** \class DirectSolver
 *
 * \brief Implementation of a direct solver to be used on the coarsest grid for problem
 *
 * \f[
 *   A x = b
 * \f]
 *
 * It internally converts the Image and StencilImage objects to vnl objects vectors and
 * matrix in order to use the class vnl_sparse_lu; the conversion of the matrix and its
 * LU decomposition is generated and stored at construction time, and is available for
 * every subsequent step to avoid unnecessary computations.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class ITK_EXPORT DirectSolver
{
public:

  /** Standard class typedefs. */
  typedef double                                                                 Precision;
  typedef DirectSolver                                                           Self;

  typedef Neighborhood< Precision, VDimension >                                  StencilType;
  typedef StencilImage< Precision, VDimension >                                  StencilImageType;

  typedef Image< Precision, VDimension >                                         ImageType;

  typedef ImageRegion< VDimension >                                              ImageRegionType;
  typedef typename ImageType::SpacingType                                        SpacingType;
  typedef typename ImageType::SizeType                                           SizeType;
  typedef typename ImageType::IndexType                                          IndexType;
  typedef typename ImageType::OffsetType                                         OffsetType;

  typedef vnl_sparse_lu                                                          SolverType;

  /** Method returning the solution \f$ x \f$ of problem \f$ A x = b\f$, where rhsImage
   *  corresponds to term \f$ b \f$. */
  typename ImageType::Pointer Solve( const ImageType * rhsImage ) const;

  /** Class constructor, which converts matrixImage to the matrix \f$ A \f$
   *  and computes its LU decomposition. */
  DirectSolver( const StencilImageType * matrixImage );

  /** Class destructor. */
  ~DirectSolver() {};

private:

  /** Utility function to manage the transfer from images to vectors and
   * vice versa. It returns the lexicographic position of index in a region
   * characterized by size. */
  inline unsigned int LexPosition( const SizeType size, const IndexType index ) const
  {

    unsigned int pos = index[ VDimension - 1 ];

    for ( int i = VDimension - 1; i > 0; --i )
      pos = pos * size[ i - 1 ] + index[ i - 1 ];

    return pos;

  }

  SolverType *                                  m_Solver;


  /** Purposely not implemented */
  DirectSolver( const Self & );
  Self & operator= ( const Self & );

};


} // end namespace mad

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDirectSolver.hxx"
#endif

#endif /* __itkDirectSolver_h */
