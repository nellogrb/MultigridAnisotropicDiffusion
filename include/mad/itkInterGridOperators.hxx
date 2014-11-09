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

#ifndef __itkInterGridOperators_hxx
#define __itkInterGridOperators_hxx

#include <list>

#include "itkNeighborhoodAlgorithm.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkInterGridOperators.h"

namespace itk
{

namespace mad
{

template < unsigned int VDimension >
InterGridOperators< VDimension >
::InterGridOperators( const std::array< CoarseGridCenteringType, VDimension > & centering ):
  m_Centering( centering )
{


}


template < unsigned int VDimension >
typename InterGridOperators< VDimension >::ImageType::Pointer
InterGridOperators< VDimension >
::Interpolation( const ImageType * inputImage ) const
{

  // Computing useful variables
  ImageRegionType inputRegion = inputImage->GetLargestPossibleRegion();
  SizeType inputSize = inputRegion.GetSize();
  IndexType inputIndex = inputRegion.GetIndex();
  SpacingType inputSpacing = inputImage->GetSpacing();

  SizeType outputSize;
  IndexType outputIndex = inputIndex;
  SpacingType outputSpacing;

  for ( unsigned int d = 0; d < VDimension; ++d )
    {

      outputSpacing[ d ] = inputSpacing[ d ] / 2;
      if ( m_Centering[ d ] == vertex ) outputSize[ d ] = ( inputSize[ d ] - 1 ) * 2 + 1;
      else outputSize[ d ] = inputSize[ d ] * 2;

    }

  ImageRegionType outputRegion( outputIndex, outputSize );

  // Preparing output image
  typename ImageType::Pointer outputImage = ImageType::New();

  outputImage->SetRegions( outputRegion );
  outputImage->Allocate();
  outputImage->SetSpacing( outputSpacing );
  outputImage->FillBuffer( 0. );

  // Generating a Stencil corresponding to a generic interior point
  // of the image, and creating a list of the non-zero entries
  std::array< PointPositionType, VDimension > pointPosition;
  for ( unsigned int d = 0; d < VDimension; ++d)
    pointPosition[ d ] = interior;

  StencilType interiorStencil = this->GenerateStencil( pointPosition, interpolationVertexStencils1D, interpolationCellStencils1D );

  std::list< OffsetType > activeOffsets;
  for ( unsigned int off = 0; off < interiorStencil.Size(); ++off)
      if ( interiorStencil[ off ] != 0. ) activeOffsets.push_back( interiorStencil.GetOffset( off ) );


  // Splitting the image region in interior and border regions, to
  // avoid unnecessary computations when in the first case
  typedef itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< ImageType > FaceCalculatorType;
  FaceCalculatorType faceCalculator;
  typename FaceCalculatorType::FaceListType faceList;
  SizeType borderSize;
  borderSize.Fill( 1 );
  faceList = faceCalculator( inputImage, inputRegion, borderSize);
  typename FaceCalculatorType::FaceListType::iterator faceIterator = faceList.begin();

  ImageRegionConstIteratorWithIndex< ImageType > inputIterator( inputImage, * faceIterator);

  // Interior points cycle
  while ( !inputIterator.IsAtEnd() )
    {

    inputIndex = inputIterator.GetIndex();

    for ( unsigned int d = 0; d < VDimension; ++d )
      outputIndex[ d ] = inputIndex[ d ] * 2;

    for ( typename std::list< OffsetType >::const_iterator activeOffsetsIterator = activeOffsets.begin();
          activeOffsetsIterator != activeOffsets.end(); ++activeOffsetsIterator )
      {

      outputImage->GetPixel( outputIndex + ( * activeOffsetsIterator ) ) +=
          interiorStencil[ * activeOffsetsIterator ] * inputImage->GetPixel( inputIndex );

      }

    ++inputIterator;

    }


  // Border points cycle
  for ( ++faceIterator; faceIterator != faceList.end(); ++faceIterator)
    {

    ImageRegionConstIteratorWithIndex< ImageType > inputBorderIterator( inputImage, * faceIterator);

    while ( !inputBorderIterator.IsAtEnd() )
      {

      inputIndex = inputBorderIterator.GetIndex();

      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        outputIndex[ d ] = inputIndex[ d ] * 2;

        pointPosition[ d ] = interior;

        if ( inputIndex[ d ] == 0 ) pointPosition[ d ] = left;
        else if ( inputRegion.GetSize( d ) - inputIndex[ d ] == 1 ) pointPosition[ d ] = right;

        }

      StencilType borderStencil = this->GenerateStencil( pointPosition, interpolationVertexStencils1D, interpolationCellStencils1D );

      for ( typename std::list< OffsetType >::const_iterator activeOffsetsIterator = activeOffsets.begin();
            activeOffsetsIterator != activeOffsets.end(); ++activeOffsetsIterator )
        {

        if ( outputRegion.IsInside( outputIndex + ( * activeOffsetsIterator ) ) )
            outputImage->GetPixel( outputIndex + ( * activeOffsetsIterator ) ) +=
                borderStencil[ * activeOffsetsIterator ] * inputImage->GetPixel( inputIndex );

        }

      ++inputBorderIterator;

      }

    }


  return outputImage;

}


template < unsigned int VDimension >
typename InterGridOperators< VDimension >::ImageType::Pointer
InterGridOperators< VDimension >
::Restriction( const ImageType * inputImage ) const
{

  // Computing useful variables
  ImageRegionType inputRegion = inputImage->GetLargestPossibleRegion();
  SizeType inputSize = inputRegion.GetSize();
  IndexType inputIndex = inputRegion.GetIndex();
  SpacingType inputSpacing = inputImage->GetSpacing();

  SizeType outputSize;
  IndexType outputIndex = inputIndex;
  SpacingType outputSpacing;

  for ( unsigned int d = 0; d < VDimension; ++d )
    {

      outputSpacing[ d ] = inputSpacing[ d ] * 2;
      if ( m_Centering[ d ] == vertex ) outputSize[ d ] = ( inputSize[ d ] - 1 ) / 2 + 1;
      else outputSize[ d ] = inputSize[ d ] / 2;

    }

  ImageRegionType outputRegion( outputIndex, outputSize );

  // Preparing output image
  typename ImageType::Pointer outputImage = ImageType::New();

  outputImage->SetRegions( outputRegion );
  outputImage->Allocate();
  outputImage->SetSpacing( outputSpacing );
  outputImage->FillBuffer( 0. );

  // Generating a Stencil corresponding to a generic interior point
  // of the image, and creating a list of the non-zero entries
  std::array< PointPositionType, VDimension > pointPosition;
  for ( unsigned int d = 0; d < VDimension; ++d)
    pointPosition[ d ] = interior;

  StencilType interiorStencil = this->GenerateStencil( pointPosition, restrictionVertexStencils1D, restrictionCellStencils1D );

  std::list< OffsetType > activeOffsets;
  for ( unsigned int off = 0; off < interiorStencil.Size(); ++off)
      if ( interiorStencil[ off ] != 0. ) activeOffsets.push_back( interiorStencil.GetOffset( off ) );

  // Splitting the image region in interior and border regions, to
  // avoid unnecessary computations when in the first case
  typedef itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< ImageType > FaceCalculatorType;
  FaceCalculatorType faceCalculator;
  typename FaceCalculatorType::FaceListType faceList;
  SizeType borderSize;
  borderSize.Fill( 1 );
  faceList = faceCalculator( outputImage, outputRegion, borderSize );
  typename FaceCalculatorType::FaceListType::iterator faceIterator = faceList.begin();


  ImageRegionIteratorWithIndex< ImageType > outputIterator( outputImage, * faceIterator );

  // Interior points cycle
  while ( !outputIterator.IsAtEnd() )
    {

    outputIndex = outputIterator.GetIndex();

    for ( unsigned int d = 0; d < VDimension; ++d )
      inputIndex[ d ] = outputIndex[ d ] * 2;

    Precision value = 0.;
    for ( typename std::list< OffsetType >::const_iterator activeOffsetsIterator = activeOffsets.begin();
          activeOffsetsIterator != activeOffsets.end(); ++activeOffsetsIterator )
      {

      value += interiorStencil[ * activeOffsetsIterator ] * inputImage->GetPixel( inputIndex + ( * activeOffsetsIterator ) );

      }

    outputIterator.Set( value );

    ++outputIterator;

    }


  // Border points cycle
  for ( ++faceIterator; faceIterator != faceList.end(); ++faceIterator)
    {

    ImageRegionIteratorWithIndex< ImageType > outputBorderIterator( outputImage, * faceIterator);

    while ( !outputBorderIterator.IsAtEnd() )
      {

      outputIndex = outputBorderIterator.GetIndex();

      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        inputIndex[ d ] = outputIndex[ d ] * 2;

        pointPosition[ d ] = interior;

        if ( outputIndex[ d ] == 0 ) pointPosition[ d ] = left;
        else if ( outputSize[ d ] - outputIndex[ d ] == 1 ) pointPosition[ d ] = right;

        }

      StencilType borderStencil = this->GenerateStencil( pointPosition, restrictionVertexStencils1D, restrictionCellStencils1D );

      Precision value = 0.;
      for ( typename std::list< OffsetType >::const_iterator activeOffsetsIterator = activeOffsets.begin();
            activeOffsetsIterator != activeOffsets.end(); ++activeOffsetsIterator )
        {
          if ( inputRegion.IsInside( inputIndex + ( * activeOffsetsIterator ) ) )
            value += borderStencil[ * activeOffsetsIterator ]
                     * inputImage->GetPixel( inputIndex + ( * activeOffsetsIterator ) );

        }

      outputBorderIterator.Set( value );

      ++outputBorderIterator;
      }

    }


  return outputImage;
}


template < unsigned int VDimension >
typename InterGridOperators< VDimension >::StencilType
InterGridOperators< VDimension >
::GenerateStencil( const std::array< PointPositionType, VDimension > & pointPosition,
                   const std::map< PointPositionType, std::array< Precision, 3 > > & vertexStencils1D,
                   const std::map< PointPositionType, std::array< Precision, 5 > > & cellStencils1D ) const
{

  // Computing the resulting stencil radius
  SizeType stencilRadius;

  for ( unsigned int d = 0; d < VDimension; ++d )
    ( m_Centering[ d ] == vertex ) ? stencilRadius[ d ] = 1 : stencilRadius[ d ] = 2;

  StencilType outputStencil;
  outputStencil.SetRadius( stencilRadius );

  OffsetType offsetFromOrigin;
  Precision value;

  // The stencil entries are obtained with a composition
  // of the monodimensional stencils
  for ( unsigned int i = 0; i < outputStencil.Size(); ++i )
    {

    value = 1;
    offsetFromOrigin = outputStencil.GetOffset( i ) + stencilRadius;

    for ( unsigned int d = 0; d < VDimension; ++d )
      {

        if ( m_Centering[ d ] == vertex )
          value *= ( vertexStencils1D.at( pointPosition[ d ] ) )[ offsetFromOrigin[ d ] ];

        else
          value *= ( cellStencils1D.at( pointPosition[ d ] ) )[ offsetFromOrigin[ d ] ];

      }

    outputStencil[ i ] = value;


  }

  return outputStencil;

}


} // end namespace mad

} // end namespace itk

#endif  /* __itkInterGridOperators_hxx */
