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

#ifndef __itkMultigridGaussSeidelSmoother_hxx
#define __itkMultigridGaussSeidelSmoother_hxx

#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"

#include "itkMultigridGaussSeidelSmoother.h"

namespace itk
{

namespace mad
{

template < unsigned int VDimension >
typename MultigridGaussSeidelSmoother< VDimension >::ImageType::Pointer
MultigridGaussSeidelSmoother< VDimension >
::SingleIteration( const ImageType * inputImage, const ImageType * rhsImage, const StencilImageType * matrixImage ) const
{

  // Gathering useful informations from the arguments
  ImageRegionType imageRegion = inputImage->GetLargestPossibleRegion();
  SpacingType imageSpacing = inputImage->GetSpacing();

  typename ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions( imageRegion );
  outputImage->Allocate();
  outputImage->SetSpacing( imageSpacing );

  ImageRegionConstIteratorWithIndex< ImageType > rhsIterator( rhsImage, imageRegion );
  ImageRegionConstIterator< StencilImageType > matrixIterator( matrixImage, imageRegion );

  SizeType diffusionRadius = matrixImage->GetRadius();

  ConstNeighborhoodIterator< ImageType > inputIterator( diffusionRadius, inputImage, imageRegion );
  NeighborhoodIterator< ImageType > outputIterator( diffusionRadius, outputImage, imageRegion );

  OffsetType center;
  OffsetType offset;
  center.Fill( 0 );

  Precision value;
  IndexType position;

  OffsetListType activeOffsets = matrixImage->GetActiveOffsetList();


  // Cycling through region points
  while ( !inputIterator.IsAtEnd() )
    {

    value = rhsIterator.Get();
    position = rhsIterator.GetIndex();

    // Cycling through the matrix active offsets
    for ( typename OffsetListType::iterator k = activeOffsets.begin(); k != activeOffsets.end(); ++k )
      {

      offset = * k;

      if ( imageRegion.IsInside( position + offset ) )
        {

        if ( this->LexOrder( offset, center ) )
          {

          value -= matrixIterator.Value()[ offset ] * outputIterator.GetPixel( offset );

          }
        else if ( this->LexOrder( center, offset ) )
          {

          value -= matrixIterator.Value()[ offset ] * inputIterator.GetPixel( offset );

          }

        }

      }

    outputIterator.SetCenterPixel( value / matrixIterator.Value()[ center ] );

    ++inputIterator;
    ++outputIterator;
    ++matrixIterator;
    ++rhsIterator;

    }


  return outputImage;

}


template < unsigned int VDimension >
typename MultigridGaussSeidelSmoother< VDimension >::ImageType::Pointer
MultigridGaussSeidelSmoother< VDimension >
::ComputeResidual( const ImageType * inputImage, const ImageType * rhsImage, const StencilImageType * matrixImage ) const
{

  // Gathering useful informations from the arguments
  ImageRegionType imageRegion = inputImage->GetLargestPossibleRegion();
  SpacingType imageSpacing = inputImage->GetSpacing();

  typename ImageType::Pointer residualImage = ImageType::New();
  residualImage->SetRegions( imageRegion );
  residualImage->Allocate();
  residualImage->SetSpacing( imageSpacing );

  ImageRegionConstIteratorWithIndex< ImageType > rhsIterator( rhsImage, imageRegion );
  ImageRegionConstIterator< StencilImageType > matrixIterator( matrixImage, imageRegion );

  SizeType diffusionRadius = matrixImage->GetRadius();

  ConstNeighborhoodIterator< ImageType > inputIterator( diffusionRadius, inputImage, imageRegion );
  NeighborhoodIterator< ImageType > outputIterator( diffusionRadius, residualImage, imageRegion );

  OffsetType center;
  OffsetType offset;
  center.Fill( 0 );

  Precision value;
  IndexType position;

  OffsetListType activeOffsets = matrixImage->GetActiveOffsetList();


  // Cycling through region points
  while ( !inputIterator.IsAtEnd() )
    {

    value = rhsIterator.Get();
    position = rhsIterator.GetIndex();

    // Cycling through the matrix active offsets
    for ( typename OffsetListType::iterator k = activeOffsets.begin(); k != activeOffsets.end(); ++k )
      {

      offset = * k;

      if ( imageRegion.IsInside( position + offset ) )
        {

        value -= matrixIterator.Value()[ offset ] * inputIterator.GetPixel( offset );

        }

      }

    outputIterator.SetCenterPixel( value );

    ++inputIterator;
    ++outputIterator;
    ++matrixIterator;
    ++rhsIterator;

    }

  return residualImage;

}


template < unsigned int VDimension >
MultigridGaussSeidelSmoother< VDimension >
::MultigridGaussSeidelSmoother() : MultigridSmoother< VDimension >::MultigridSmoother()
{

}


template < unsigned int VDimension >
MultigridGaussSeidelSmoother< VDimension >
::~MultigridGaussSeidelSmoother()
{

}


} // end namespace mad

} // end namespace itk

#endif  /* __itkMultigridGaussSeidelSmoother_hxx */
