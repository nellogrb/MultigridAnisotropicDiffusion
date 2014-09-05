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

#ifndef __itkDirectSolver_hxx
#define __itkDirectSolver_hxx

#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkDirectSolver.h"

namespace itk
{

namespace mad
{

template < unsigned int VDimension >
DirectSolver< VDimension >
::DirectSolver( const StencilImageType * matrixImage ) : m_Solver( nullptr )
{

  // Calculating number of points and creating the matrix
  ImageRegionType imageRegion = matrixImage->GetLargestPossibleRegion();
  SizeType imageSize = imageRegion.GetSize();

  unsigned int N = 1;
  for ( unsigned int d = 0; d < VDimension; ++d ) N *= imageRegion.GetSize( d );

  vnl_sparse_matrix< Precision > matrix( N, N );

  IndexType positionRow;
  IndexType positionColumn;

  unsigned int row;
  unsigned int column;

  ImageRegionConstIteratorWithIndex< StencilImageType > matrixIterator( matrixImage, imageRegion );

  // Copying matrixImage to vnl matrix
  while ( !matrixIterator.IsAtEnd() )
    {

    positionRow = matrixIterator.GetIndex();
    row = LexPosition( imageSize, positionRow );

    for ( unsigned int i = 0; i < matrixIterator.Value().Size(); ++i )
     {

     positionColumn = positionRow + matrixIterator.Value().GetOffset( i );

     if ( imageRegion.IsInside( positionColumn ) )
       {

       column = LexPosition( imageSize, positionColumn );

       matrix( row, column ) = matrixIterator.Value()[ i ];


       }

     }

   ++matrixIterator;
   }

  m_Solver = new vnl_sparse_lu( matrix );

  // Computing LU decomposition
  vnl_vector< Precision > rhs( N, 0 );

  m_Solver->solve( rhs );

}


template < unsigned int VDimension >
typename DirectSolver< VDimension >::ImageType::Pointer
DirectSolver< VDimension >
::Solve( const ImageType * rhsImage ) const
{

  ImageRegionType imageRegion = rhsImage->GetLargestPossibleRegion();
  SizeType imageSize = imageRegion.GetSize();

  typename ImageType::Pointer solutionImage = ImageType::New();
  solutionImage->SetRegions( imageRegion );
  solutionImage->Allocate();
  solutionImage->SetSpacing( rhsImage->GetSpacing() );

  unsigned int N = 1;
  for ( unsigned int d = 0; d < VDimension; ++d ) N *= imageRegion.GetSize( d );

  vnl_vector< Precision > solution( N );
  vnl_vector< Precision > rhs( N );

  IndexType positionRow;

  unsigned int row;

  ImageRegionConstIteratorWithIndex< ImageType > rhsIterator( rhsImage, imageRegion );

  // Copying rhsImage to vnl vector
  while ( !rhsIterator.IsAtEnd() )
    {

    positionRow = rhsIterator.GetIndex();
    row = LexPosition( imageSize, positionRow );
    rhs( row ) = rhsIterator.Value();

    ++rhsIterator;

    }

  solution = m_Solver->solve( rhs );

  ImageRegionIteratorWithIndex< ImageType > solutionIterator( solutionImage, imageRegion );

  // Copying solutionImage back to image itk format
  while ( !solutionIterator.IsAtEnd() )
    {

    positionRow = solutionIterator.GetIndex();
    row = LexPosition( imageSize, positionRow );
    solutionIterator.Value() = solution( row );

    ++solutionIterator;

    }

  return solutionImage;

}


} // end namespace mad

} // end namespace itk

#endif  /* __itkDirectSolver_hxx */
