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

#ifndef __itkCoarseGridOperatorsGenerator_hxx
#define __itkCoarseGridOperatorsGenerator_hxx

#include <array>
#include <iostream>

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNeighborhoodAlgorithm.h"

#include "itkCoarseGridOperatorsGenerator.h"

namespace itk
{

namespace mad
{

template < unsigned int VDimension >
CoarseGridOperatorsGenerator< VDimension >
::CoarseGridOperatorsGenerator( const CoarseGridOperatorType & operatorType ):
  m_CoarseGridOperator( operatorType ) {}


template < unsigned int VDimension >
void
CoarseGridOperatorsGenerator< VDimension >
::GenerateOperators( GridsHierarchy< VDimension > * gridsHierarchy, const TensorImageType * fineTensor,
                     const Precision timeStep ) const
{

  unsigned int numberOfLevels = gridsHierarchy->GetMaxDepth() + 1;

  // Generating the operator on the finest grid
  this->GenerateDCA( gridsHierarchy->GetGridAtLevel( 0 ), fineTensor, timeStep );

  switch( m_CoarseGridOperator )
    {
    case DCA:
      {

      // In the DCA case, we interpolate the diffusion tensor's coefficients
      // using InterGridOperators restriction and interpolation, then we
      // generate the operator on each grid
      std::array< std::array< typename ImageType::Pointer, VDimension >, VDimension >
              fineDiffusionCoefficients;

      for ( unsigned int d = 0; d < VDimension; ++d )
         {
         for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
           {

           fineDiffusionCoefficients[ d ][ d2 ] = ImageType::New();
           fineDiffusionCoefficients[ d ][ d2 ]->SetRegions( gridsHierarchy->GetRegionAtLevel( 0 ) );
           fineDiffusionCoefficients[ d ][ d2 ]->Allocate();

           }
         }

      // Copying the fine diffusion tensor's coefficients into an array of
      // arrays (a fixed-size matrix) of images
      ImageRegionConstIteratorWithIndex< TensorImageType > fineTensorIterator( fineTensor, gridsHierarchy->GetRegionAtLevel( 0 ) );

      while ( !fineTensorIterator.IsAtEnd() )
        {
        for ( unsigned int d = 0; d < VDimension; ++d )
          {
          for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
            {

            fineDiffusionCoefficients[ d ][ d2 ]->SetPixel( fineTensorIterator.GetIndex(), fineTensorIterator.Value()( d, d2 ) );

            }
          }

        ++fineTensorIterator;
        }

      std::array< std::array< typename ImageType::Pointer, VDimension >, VDimension >
              coarseDiffusionCoefficients;

      // Restricting the diffusion tensor's coefficients
      for ( unsigned int l = 1; l < numberOfLevels; ++l )
        {

        InterGridOperatorsType IGOperators( gridsHierarchy->GetVertexCenteringAtLevel( l ) );

        for ( unsigned int d = 0; d < VDimension; ++d )
          {
          for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
            {

            coarseDiffusionCoefficients[ d ][ d2 ] = IGOperators.Restriction( fineDiffusionCoefficients[ d ][ d2 ] );

            }
          }

        // Creating the coarse diffusion tensor and generating the
        // correspondent coarse operator
        typename TensorImageType::Pointer coarseTensor = TensorImageType::New();
        coarseTensor->SetRegions( gridsHierarchy->GetRegionAtLevel( l ) );
        coarseTensor->Allocate();

        ImageRegionConstIteratorWithIndex< TensorImageType > coarseTensorIterator( coarseTensor, gridsHierarchy->GetRegionAtLevel( l ) );

        while ( !coarseTensorIterator.IsAtEnd() )
          {
          for ( unsigned int d = 0; d < VDimension; ++d )
            {
            for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
              {

              coarseTensor->GetPixel( coarseTensorIterator.GetIndex() )( d, d2 ) = coarseDiffusionCoefficients[ d ][ d2 ]->GetPixel( coarseTensorIterator.GetIndex() );

              }
            }

          ++coarseTensorIterator;

          }

        this->GenerateDCA( gridsHierarchy->GetGridAtLevel( l ), coarseTensor, timeStep );

        // Copying back the coarse coefficients to iterate the procedure
        for ( unsigned int d = 0; d < VDimension; ++d )
          {
          for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
            {

            fineDiffusionCoefficients[ d ][ d2 ] = coarseDiffusionCoefficients[ d ][ d2 ];

            }
          }

        }

      }
      break;

    case GCA:
      {

      for ( unsigned int l = 1; l < numberOfLevels; ++l )
        {

        this->GenerateGCA( gridsHierarchy->GetGridAtLevel( l - 1 ), gridsHierarchy->GetGridAtLevel( l ) );

        }

      }
      break;
    }

}


template < unsigned int VDimension >
void
CoarseGridOperatorsGenerator< VDimension >
::GenerateDCA( typename GridsHierarchy< VDimension >::Grid * grid, const TensorImageType * tensor,
               const Precision timeStep ) const
{

  OffsetType axes[ VDimension ];
  for ( unsigned int d = 0; d < VDimension; ++d )
    {

    axes[ d ].Fill( 0 );
    axes[ d ][ d ] = 1;

    }

  grid->g_CoarseOperator = StencilImageType::New();
  grid->g_CoarseOperator->SetRegions( grid->g_Region );
  grid->g_CoarseOperator->Allocate();

  SizeType gridSize = grid->g_Region.GetSize();

  OffsetType center;
  OffsetType offsetP;
  OffsetType offsetM;
  OffsetType offsetP2;
  OffsetType offsetM2;
  OffsetType offsetPP;
  OffsetType offsetPM;
  OffsetType offsetMP;
  OffsetType offsetMM;
  center.Fill( 0 );

  Precision weight;
  Precision value;
  IndexType index;

  ImageRegionIteratorWithIndex< StencilImageType > operatorIterator( grid->g_CoarseOperator, grid->g_Region );

  while ( !operatorIterator.IsAtEnd() )
    {

    index = operatorIterator.GetIndex();
    operatorIterator.Value().SetRadius( 1 );

    // Initializing every element of the stencil in index position to 0
    for ( unsigned int i = 0; i < operatorIterator.Value().Size(); ++i ) operatorIterator.Value()[ i ] = 0;

    operatorIterator.Value()[ center ] = 1;


    // Calculating the matrix coefficients. The homogeneous Neumann border conditions
    // are taken into account by redefining the offsets coordinates (since we impose
    // that the value of a pixel outside the grid must be the same as the one of its
    // symmetric correspondent with respect to the border)
    for ( unsigned int d = 0; d < VDimension; ++d )
      {

      offsetP = center + axes[ d ];
      offsetM = center - axes[ d ];

      // Second order central difference for the second derivatives
      weight = - timeStep / ( grid->g_Spacing[ d ] * grid->g_Spacing[ d ] );

      if ( index[ d ] == 0 ) offsetM = center + axes[ d ];
      else if ( gridSize[ d ] - index[ d ] == 1 ) offsetP = center - axes[ d ];

      value = tensor->GetPixel( index )( d, d ) * weight;

      operatorIterator.Value()[ offsetP ] += value;
      operatorIterator.Value()[ offsetM ] += value;
      operatorIterator.Value()[ center ] -= 2 * value;

      for ( unsigned int d2 = 0; d2 < VDimension; ++d2 )
        {

        weight = - timeStep / ( 4 * grid->g_Spacing[ d ] * grid->g_Spacing[ d2 ] );

        offsetP2 = center + axes[ d2 ];
        offsetM2 = center - axes[ d2 ];

        offsetPP = ( center + axes[ d ] ) + axes[ d2 ];
        offsetPM = ( center + axes[ d ] );
        offsetPM -= axes[ d2 ];

        offsetMP = ( center - axes[ d ] ) + axes[ d2 ];
        offsetMM = ( center - axes[ d ] );
        offsetMM -= axes[ d2 ];


        if ( index[ d ] == 0 )
          {

          offsetMM += axes[ d ];
          offsetMM += axes[ d ];
          offsetMP += axes[ d ];
          offsetMP += axes[ d ];

          }
        else if ( gridSize[ d ] - index[ d ] == 1 )
          {

          offsetPP -= axes[ d ];
          offsetPP -= axes[ d ];
          offsetPM -= axes[ d ];
          offsetPM -= axes[ d ];

          }

        if ( index[ d2 ] == 0 )
          {

          offsetMM += axes[ d2 ];
          offsetMM += axes[ d2 ];
          offsetPM += axes[ d2 ];
          offsetPM += axes[ d2 ];

          offsetM2 += axes[ d2 ];
          offsetM2 += axes[ d2 ];

          }
        else if ( gridSize[ d2 ] - index[ d2 ] == 1 )
          {

          offsetPP -= axes[ d2 ];
          offsetPP -= axes[ d2 ];
          offsetMP -= axes[ d2 ];
          offsetMP -= axes[ d2 ];

          offsetP2 -= axes[ d2 ];
          offsetP2 -= axes[ d2 ];

          }


        // Second order central difference for the mixed derivatives
        if ( d != d2 )
          {

          value = tensor->GetPixel( index )( d, d2 ) * weight;

          operatorIterator.Value()[ offsetPP ] += value;
          operatorIterator.Value()[ offsetPM ] -= value;
          operatorIterator.Value()[ offsetMP ] -= value;
          operatorIterator.Value()[ offsetMM ] += value;

          }


        // Second order central difference for the first derivatives.
        // We also use second order backward/forward finite differences
        // for the derivative of the tensor's coefficients, if the point
        // lays on the border
        if ( index[ d2 ] == 0 )
          {

          value = ( - 3. * tensor->GetPixel( index )( d, d2 ) + 4. * tensor->GetPixel( index + axes[ d2 ] )( d, d2 )
                  - 1. * tensor->GetPixel( index + axes[ d2 ] + axes[ d2 ] )( d, d2 ) ) * weight;

          }
        else if ( gridSize[ d2 ] - index[ d2 ] == 1 )
          {

          value = ( 3. * tensor->GetPixel( index )( d, d2 ) - 4. * tensor->GetPixel( index - axes[ d2 ] )( d, d2 )
                  + 1. * tensor->GetPixel( index - axes[ d2 ] - axes[ d2 ] )( d, d2 ) ) * weight;

          }
        else
          {

          value = ( tensor->GetPixel( index + axes[ d2 ] )( d, d2 ) - tensor->GetPixel( index - axes[ d2 ] )( d, d2 ) ) * weight;

          }

          operatorIterator.Value()[ offsetP ] += value;
          operatorIterator.Value()[ offsetM ] -= value;

        }

      }


    ++operatorIterator;

    }

  // Defining diffusion radius (in this case it is always 1) and activating
  // the correspondent offsets
  SizeType diffusionRadius;
  diffusionRadius.Fill( 1 );
  grid->g_CoarseOperator->SetRadius( diffusionRadius );
  grid->g_CoarseOperator->ActivateAllOffsets();

}


template < unsigned int VDimension >
void
CoarseGridOperatorsGenerator< VDimension >
::GenerateGCA( typename GridsHierarchy< VDimension >::Grid * fineGrid,
               typename GridsHierarchy< VDimension >::Grid * coarseGrid ) const
{

  ImageRegionType fineRegion = fineGrid->g_Region;
  ImageRegionType coarseRegion = coarseGrid->g_Region;

  InterGridOperatorsType IGOperators( coarseGrid->g_VertexCentered );

  PointPositionType p[ VDimension ];
  for ( unsigned int d = 0; d < VDimension; ++d ) p[ d ] = PointPositionType::interior;



  // Creating an image composed of one central pixel of value 1, padded by
  // zeroes to correctly interpolate it, and the resulting interpolated image
  SizeType onePixelSize;
  onePixelSize.Fill( 3 );
  IndexType onePixelOrigin;
  onePixelOrigin.Fill( 0 );
  IndexType onePixelCenter;
  onePixelCenter.Fill( 1 );

  ImageRegionType onePixelRegion( onePixelOrigin, onePixelSize );

  typename ImageType::Pointer onePixelImage = ImageType::New();
  onePixelImage->SetRegions( onePixelRegion );
  onePixelImage->Allocate();
  onePixelImage->SetSpacing( coarseGrid->g_Spacing );
  onePixelImage->FillBuffer( 0. );
  onePixelImage->SetPixel( onePixelCenter, 1. );

  typename ImageType::Pointer interpolatedOnePixelImage = IGOperators.Interpolation( onePixelImage );

  ImageRegionType interpolatedOnePixelImageRegion = interpolatedOnePixelImage->GetLargestPossibleRegion();


  // Calculating the diffusion stencil's radius on the coarse grid; the first time
  // that a cell centered approach is used in a certain direction where the finer
  // diffusion stencil has a unitary radius, the coarse diffusion radius grows from
  // 1 unit to 2 units (in that direction).
  SizeType fineDiffusionRadius = fineGrid->g_CoarseOperator->GetRadius();
  SizeType coarseDiffusionRadius;
  SizeType coarseDiffusionSize;

  for ( unsigned int d = 0; d < VDimension; ++d )
    {

      if ( ( fineDiffusionRadius[ d ] == 1 && coarseGrid->g_VertexCentered[ d ] == false ) )
        coarseDiffusionRadius[ d ] = fineDiffusionRadius[ d ] + 1;
      else coarseDiffusionRadius[ d ] = fineDiffusionRadius[ d ];

      coarseDiffusionSize[ d ] = 1 + 2 * coarseDiffusionRadius[ d ];

    }


  // Creating an image which will be filled with the effect of the diffusion on
  // interpolatedOnePixelImage; we can predict its size and position relative to
  // interpolatedOnePixelImage images using the coarseDiffusionSize
  SizeType diffusedOnePixelImageSize;
  IndexType diffusedOnePixelCenter;
  OffsetType diffusedInterpolatedRelativePosition;

  for ( unsigned int d = 0; d < VDimension; ++d )
    {

    if ( coarseGrid->g_VertexCentered[ d ] == 0 ) diffusedOnePixelImageSize[ d ] = 2 * coarseDiffusionSize[ d ];
    else diffusedOnePixelImageSize[ d ] = 2 * coarseDiffusionSize[ d ] - 1;

    diffusedInterpolatedRelativePosition[ d ] = ( diffusedOnePixelImageSize[ d ] - interpolatedOnePixelImageRegion.GetSize( d ) ) / 2;

    diffusedOnePixelCenter[ d ] = ( diffusedOnePixelImageSize[ d ] - 1 ) / 2;

    }

  ImageRegionType diffusedOnePixelImageRegion( onePixelOrigin, diffusedOnePixelImageSize );

  typename ImageType::Pointer diffusedOnePixelImage = ImageType::New();
  diffusedOnePixelImage->SetRegions( diffusedOnePixelImageRegion );
  diffusedOnePixelImage->Allocate();


  // Initializing coarse grid operator
  coarseGrid->g_CoarseOperator = StencilImageType::New();
  coarseGrid->g_CoarseOperator->SetRegions( coarseGrid->g_Region );
  coarseGrid->g_CoarseOperator->Allocate();
  coarseGrid->g_CoarseOperator->SetRadius( coarseDiffusionRadius );
  coarseGrid->g_CoarseOperator->ActivateAllOffsets();

  ImageRegionIteratorWithIndex< StencilImageType >
    coarseOperatorInitializationIterator( coarseGrid->g_CoarseOperator, coarseGrid->g_Region );

  while ( !coarseOperatorInitializationIterator.IsAtEnd() )
    {

    coarseOperatorInitializationIterator.Value().SetRadius( coarseDiffusionRadius );

    for ( unsigned int i = 0; i < coarseOperatorInitializationIterator.Value().Size(); ++i )
      coarseOperatorInitializationIterator.Value()[ i ] = 0;

    ++coarseOperatorInitializationIterator;

    }


  // Splitting the region in interior and border faces, so we don't need to recompute
  // the interpolated pixel every time in the interior region, as a different
  // procedure will be needed at the border. We exclude from the first region subset
  // all the points laying at least 3 points from the border, to simplify the
  // subsequent cycle
  typedef itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< StencilImageType > FaceCalculatorType;
  FaceCalculatorType faceCalculator;
  SizeType borderWidth;
  borderWidth.Fill( 3 );
  typename FaceCalculatorType::FaceListType faceList;

  faceList = faceCalculator( coarseGrid->g_CoarseOperator, coarseGrid->g_CoarseOperator->GetLargestPossibleRegion(), borderWidth );
  typename FaceCalculatorType::FaceListType::iterator faceIterator = faceList.begin();

  ImageRegionIteratorWithIndex< StencilImageType > coarseOperatorInteriorIterator( coarseGrid->g_CoarseOperator, * faceIterator );


  // Defining variables which will be needed during the cycle
  IndexType fineIndex;
  IndexType coarseIndex;
  IndexType operatorIndex;

  IndexType neighborIndex;
  OffsetType tempOffset;
  OffsetListType fineActiveOffsets = fineGrid->g_CoarseOperator->GetActiveOffsetList();
  OffsetListType coarseActiveOffsets = coarseGrid->g_CoarseOperator->GetActiveOffsetList();

  Precision value;

  bool ignoreLeftBorder[ VDimension ];
  bool ignoreRightBorder[ VDimension ];


  // Interior cycle over coarseOperator's interior pixels
  while ( !coarseOperatorInteriorIterator.IsAtEnd() )
    {

    // Getting the index on the coarse grid and its correspondent
    // on the fine grid, and initializing diffusedOnePixelImage
    coarseIndex = coarseOperatorInteriorIterator.GetIndex();
    for ( unsigned int d = 0; d < VDimension; ++d ) fineIndex[ d ] = 2 * coarseIndex[ d ];

    diffusedOnePixelImage->FillBuffer( 0. );

    ImageRegionIteratorWithIndex< ImageType > diffusedIterator( diffusedOnePixelImage, diffusedOnePixelImageRegion );

    // Iterating through diffusedOnePixelImage pixels to compute their value
    while( !diffusedIterator.IsAtEnd() )
      {

      value = 0;

      // For each point of diffusedOnePixelImage, we iterate once more through
      // the values of the correspondent Neighborhood of the operator on the
      // fine grid (using only the active offsets)
      for ( typename OffsetListType::iterator k = fineActiveOffsets.begin(); k != fineActiveOffsets.end(); ++k )
        {

        tempOffset = * k;

        // Converting the position of the neighbor to the correct one from
        // interpolatedOnePixelImage point of view
        neighborIndex = diffusedIterator.GetIndex() + tempOffset;
        neighborIndex -= diffusedInterpolatedRelativePosition;

        for ( unsigned int d = 0; d < VDimension; ++d )
          operatorIndex[ d ] = diffusedOnePixelCenter[ d ] - diffusedIterator.GetIndex()[ d ] + fineIndex[ d ];

        if ( interpolatedOnePixelImageRegion.IsInside( neighborIndex ) )
          {

          value += fineGrid->g_CoarseOperator->GetPixel( operatorIndex )[ tempOffset ]
                   * interpolatedOnePixelImage->GetPixel( neighborIndex );

          }

        }

      diffusedIterator.Set( value );

      ++diffusedIterator;

      }

    for ( unsigned int d = 0; d < VDimension; ++d )
      {

      ignoreLeftBorder[ d ] = true;
      ignoreRightBorder[ d ] = true;

      }


    // Restricting the resulting diffusedOnePixelImage. The second and third
    // arguments of Restriction are arrays filled with 'true', since the points on
    // the border of diffusedOnePixelImage do not really belong to the border
    // and hence have to be treated as interior points
    typename ImageType::Pointer restrictedOnePixelImage = IGOperators.Restriction( diffusedOnePixelImage, ignoreLeftBorder, ignoreRightBorder );

    OffsetType restrictedCenter;

    for ( unsigned int d = 0; d < VDimension; ++d ) restrictedCenter[ d ] = coarseDiffusionRadius[ d ];

    ImageRegionIteratorWithIndex< ImageType > restrictedIterator( restrictedOnePixelImage, restrictedOnePixelImage->GetLargestPossibleRegion() );

    // Copying the values into the coarse operator
    while ( !restrictedIterator.IsAtEnd() )
      {

      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        operatorIndex[ d ] = restrictedIterator.GetIndex()[ d ] + coarseIndex[ d ] - coarseDiffusionRadius[ d ];
        tempOffset[ d ] = coarseDiffusionRadius[ d ] - restrictedIterator.GetIndex()[ d ];

        }

      coarseGrid->g_CoarseOperator->GetPixel( operatorIndex )[ tempOffset ] = restrictedIterator.Value();

      ++restrictedIterator;

      }


    ++coarseOperatorInteriorIterator;

    }




  // Different cycles over coarseOperator's border pixels (and things get really complicated)
  for ( ++faceIterator; faceIterator != faceList.end(); ++faceIterator)
    {

    ImageRegionIteratorWithIndex< StencilImageType > coarseOperatorBorderIterator( coarseGrid->g_CoarseOperator, * faceIterator );

    while ( !coarseOperatorBorderIterator.IsAtEnd() )
      {

      coarseIndex = coarseOperatorBorderIterator.GetIndex();

      IndexType onePixelPosition;

      // We still have to differentiate between border points and 'almost' border points,
      // since we excluded them in the first cycle. Because of this, we still have to deal
      // with 3 possible positions for each direction (and for each side). Now we think of
      // onePixelImage as aligned with the border, instead of being centered on the iterating
      // index as in the previous cycle
      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        fineIndex[ d ] = 2 * coarseIndex[ d ];

        ignoreLeftBorder[ d ] = true;
        ignoreRightBorder[ d ] = true;

        if ( coarseIndex[ d ] < 3 ) ignoreLeftBorder[ d ] = false;
        else if ( coarseRegion.GetSize( d ) - coarseIndex[ d ] < 4 ) ignoreRightBorder[ d ] = false;

        if ( coarseIndex[ d ] == 0 || coarseRegion.GetSize( d ) - coarseIndex[ d ] == 3 ) onePixelPosition[ d ] = 0;
        else if ( coarseIndex[ d ] == 2 || coarseRegion.GetSize( d ) - coarseIndex[ d ] == 1 ) onePixelPosition[ d ] = 2;
        else onePixelPosition[ d ] = 1;

        }

      onePixelImage->FillBuffer( 0. );
      onePixelImage->SetPixel( onePixelPosition, 1. );

      typename ImageType::Pointer interpolatedBorderOnePixelImage = IGOperators.Interpolation( onePixelImage, ignoreLeftBorder, ignoreRightBorder );
      ImageRegionType interpolatedOnePixelRegion = interpolatedBorderOnePixelImage->GetLargestPossibleRegion();

      SizeType borderCoarseDiffusionSize;

      // Now the diffusedOnePixelImageSize is not fixed anymore and
      // depends on the pixel position, we define it so that one of
      // its sides is aligned with the border
      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        if ( coarseIndex[ d ] < 3 ) borderCoarseDiffusionSize[ d ] = coarseDiffusionRadius[ d ] + coarseIndex[ d ] + 1;
        else if ( coarseRegion.GetSize( d ) - coarseIndex[ d ] < 4 )
          borderCoarseDiffusionSize[ d ] = coarseDiffusionRadius[ d ] + ( coarseRegion.GetSize( d ) - coarseIndex[ d ] );
        else borderCoarseDiffusionSize[ d ] = coarseDiffusionSize[ d ];


        if ( coarseGrid->g_VertexCentered[ d ] == false ) diffusedOnePixelImageSize[ d ] = 2 * borderCoarseDiffusionSize[ d ];
        else diffusedOnePixelImageSize[ d ] = 2 * borderCoarseDiffusionSize[ d ] - 1;


        if ( coarseIndex[ d ] < 3 ) diffusedInterpolatedRelativePosition[ d ] = 0;
        else if ( coarseRegion.GetSize( d ) - coarseIndex[ d ] < 4 )
          diffusedInterpolatedRelativePosition[ d ] = diffusedOnePixelImageSize[ d ] - interpolatedOnePixelRegion.GetSize( d );
        else
          diffusedInterpolatedRelativePosition[ d ] = ( diffusedOnePixelImageSize[ d ] - interpolatedOnePixelRegion.GetSize( d ) ) / 2;

        }

      ImageRegionType diffusedBorderOnePixelImageRegion( onePixelOrigin, diffusedOnePixelImageSize );

      typename ImageType::Pointer diffusedBorderOnePixelImage = ImageType::New();

      diffusedBorderOnePixelImage->SetRegions( diffusedBorderOnePixelImageRegion );
      diffusedBorderOnePixelImage->Allocate();
      diffusedBorderOnePixelImage->FillBuffer( 0. );

      ImageRegionIterator< ImageType > diffusedIterator( diffusedBorderOnePixelImage, diffusedBorderOnePixelImageRegion );


      // As before, we iterate through the values of diffusedOnePixelImage
      while( !diffusedIterator.IsAtEnd() )
        {

        value = 0;

        for ( typename OffsetListType::iterator k = fineActiveOffsets.begin(); k != fineActiveOffsets.end(); ++k )
          {

          tempOffset = * k;

          // Converting the position of the neighbor to the correct one from
          // interpolatedOnePixelImage point of view
          neighborIndex = diffusedIterator.GetIndex() + tempOffset;
          neighborIndex -= diffusedInterpolatedRelativePosition;

          for ( unsigned int d = 0; d < VDimension; ++d )
            {

            if ( coarseIndex[ d ] < 3 ) operatorIndex[ d ] = diffusedIterator.GetIndex()[ d ];
            else if ( coarseRegion.GetSize( d ) - coarseIndex[ d ] < 4 )
              operatorIndex[ d ] = fineRegion.GetSize( d ) - diffusedOnePixelImageSize[ d ] + diffusedIterator.GetIndex()[ d ];
            else operatorIndex[ d ] = diffusedOnePixelCenter[ d ] - diffusedIterator.GetIndex()[ d ] + fineIndex[ d ];

            }

          if ( interpolatedOnePixelImageRegion.IsInside( neighborIndex ) )
            {

            value += fineGrid->g_CoarseOperator->GetPixel( operatorIndex )[ tempOffset ]
                     * interpolatedBorderOnePixelImage->GetPixel( neighborIndex );

            }

          }


        diffusedIterator.Set( value );

        ++diffusedIterator;

        }


      // Restricting diffusedBorderOnePixelImage
      typename ImageType::Pointer restrictedOnePixelImage = IGOperators.Restriction( diffusedBorderOnePixelImage, ignoreLeftBorder, ignoreRightBorder );

      SizeType restrictedSize = restrictedOnePixelImage->GetLargestPossibleRegion().GetSize();
      OffsetType restrictedCornerPosition;
      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        if ( coarseIndex[ d ] < 3 ) restrictedCornerPosition[ d ] = 0;
        else if ( coarseRegion.GetSize( d ) - coarseIndex[ d ] < 4 )
          restrictedCornerPosition[ d ] = coarseRegion.GetSize( d ) - restrictedSize[ d ];
        else restrictedCornerPosition[ d ] = coarseIndex[ d ] - coarseDiffusionRadius[ d ];

        }

      ImageRegionIteratorWithIndex< ImageType > restrictedIterator( restrictedOnePixelImage, restrictedOnePixelImage->GetLargestPossibleRegion() );


      // Copying the values into the coarse operator
      while ( !restrictedIterator.IsAtEnd() )
        {

        for ( unsigned int d = 0; d < VDimension; ++d )
          {

          operatorIndex[ d ] = restrictedIterator.GetIndex()[ d ] + restrictedCornerPosition[ d ];
          tempOffset[ d ] = coarseIndex[ d ] - operatorIndex[ d ];

          }

        if ( coarseRegion.IsInside( operatorIndex ) && restrictedIterator.Value() != 0 )
          {

          coarseGrid->g_CoarseOperator->GetPixel( operatorIndex )[ tempOffset ] = restrictedIterator.Value();

          }

        ++restrictedIterator;

        }


      ++coarseOperatorBorderIterator;

      }

    }

}


} // end namespace mad

} // end namespace itk

#endif  /* __itkCoarseGridOperatorsGenerator_hxx */
