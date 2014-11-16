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

#ifndef __itkGridsHierarchy_hxx
#define __itkGridsHierarchy_hxx

#include "itkGridsHierarchy.h"

namespace itk
{

namespace mad
{

template < unsigned int VDimension >
GridsHierarchy< VDimension >
::GridsHierarchy( const ImageRegionType & initialRegion, const SpacingType & initialSpacing, const TensorImageType * fineTensor,
    const Precision timeStep )
{

  // Computing max depth (we stop when the shortest side has at least 6 pixels)
  unsigned long int gridSize[ VDimension ];
  for ( unsigned int d = 0; d < VDimension; ++d ) gridSize[ d ] = initialRegion.GetSize( d );

  bool coarsestGrid = false;
  unsigned int numberOfLevels = 1;

  while ( !coarsestGrid )
    {
     for ( unsigned int d = 0; d < VDimension; ++d )
       {

       gridSize[ d ] = ( gridSize[ d ] % 2 == 0 ) ? gridSize[ d ] / 2 : ( ( gridSize[ d ] - 1 ) / 2 ) + 1;

       if ( gridSize[ d ] < 6 ) coarsestGrid = true;

       }

     ++numberOfLevels;
    }

  --numberOfLevels;

  m_MaxDepth = numberOfLevels - 1;
  m_GridLevels = new Grid[ numberOfLevels ];


  // Storing informations about the finest grid
  m_GridLevels[ 0 ].g_Region = initialRegion;
  m_GridLevels[ 0 ].g_Spacing = initialSpacing;
  m_GridLevels[ 0 ].g_CoarseOperator = 0;
  m_GridLevels[ 0 ].g_Centering.fill( CoarseGridCenteringType::vertex );

  SizeType coarseRegionSize;
  IndexType coarseRegionOrigin;


  // Computing and storing informations about the coarser grids
  for ( unsigned int l = 1; l < m_MaxDepth + 1; ++l )
    {

      for ( unsigned int d = 0; d < VDimension; ++d )
        {

        m_GridLevels[ l ].g_Spacing[ d ] = m_GridLevels[ l - 1 ].g_Spacing[ d ] * 2;

          coarseRegionOrigin[ d ] = m_GridLevels[ l - 1 ].g_Region.GetIndex( d );

          if ( this->m_GridLevels[ l - 1 ].g_Region.GetSize( d ) % 2 == 0 )
            {

              coarseRegionSize[ d ] = m_GridLevels[ l - 1 ].g_Region.GetSize( d ) / 2;
              m_GridLevels[ l ].g_Centering[ d ] = CoarseGridCenteringType::cell;

            }
          else
            {

              coarseRegionSize[ d ] = ( m_GridLevels[ l - 1 ].g_Region.GetSize( d ) - 1 ) / 2 + 1;
              m_GridLevels[ l ].g_Centering[ d ] = CoarseGridCenteringType::vertex;

            }

        }

      ImageRegionType coarseRegion( coarseRegionOrigin, coarseRegionSize );
      m_GridLevels[ l ].g_Region = coarseRegion;

      m_GridLevels[ l ].g_CoarseOperator = 0;

    }


  // Generating the operator on the finest grid
  this->GenerateDCA( m_GridLevels, fineTensor, timeStep );

  std::array< std::array< typename ImageType::Pointer, VDimension >, VDimension > fineDiffusionCoefficients;

  for ( unsigned int d = 0; d < VDimension; ++d )
     {
     for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
       {

       fineDiffusionCoefficients[ d ][ d2 ] = ImageType::New();
       fineDiffusionCoefficients[ d ][ d2 ]->SetRegions( m_GridLevels[ 0 ].g_Region );
       fineDiffusionCoefficients[ d ][ d2 ]->Allocate();

       }
     }

  // Copying the fine diffusion tensor's coefficients into an array of
  // arrays (a fixed-size matrix) of images
  ImageRegionConstIteratorWithIndex< TensorImageType > fineTensorIterator( fineTensor, m_GridLevels[ 0 ].g_Region );

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

    InterGridOperatorsType IGOperators( m_GridLevels[ l ].g_Centering );

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
    coarseTensor->SetRegions( m_GridLevels[ l ].g_Region );
    coarseTensor->Allocate();

    ImageRegionConstIteratorWithIndex< TensorImageType > coarseTensorIterator( coarseTensor, m_GridLevels[ l ].g_Region );

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

    this->GenerateDCA( &m_GridLevels[ l ], coarseTensor, timeStep );

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

template < unsigned int VDimension >
GridsHierarchy< VDimension >
::~GridsHierarchy()
{

  delete[] m_GridLevels;

}


template < unsigned int VDimension >
unsigned int
GridsHierarchy< VDimension >
::GetMaxDepth() const
{

  return m_MaxDepth;

}


template < unsigned int VDimension >
typename GridsHierarchy< VDimension >::Grid *
GridsHierarchy< VDimension >
::GetGridAtLevel( const unsigned int l )
{

  return & ( m_GridLevels[ l ] );

}


template < unsigned int VDimension >
typename GridsHierarchy< VDimension >::ImageRegionType
GridsHierarchy< VDimension >
::GetRegionAtLevel( const unsigned int l ) const
{

  return m_GridLevels[ l ].g_Region;

}


template < unsigned int VDimension >
typename GridsHierarchy< VDimension >::SpacingType
GridsHierarchy< VDimension >
::GetSpacingAtLevel( const unsigned int l ) const
{

  return m_GridLevels[ l ].g_Spacing;

}


template < unsigned int VDimension >
typename GridsHierarchy< VDimension >::StencilImageType::Pointer
GridsHierarchy< VDimension >
::GetCoarseOperatorAtLevel( const unsigned int l ) const
{

  return m_GridLevels[ l ].g_CoarseOperator;

}


template < unsigned int VDimension >
std::array< typename GridsHierarchy< VDimension >::CoarseGridCenteringType, VDimension >
GridsHierarchy< VDimension >
::GetVertexCenteringAtLevel( const unsigned int l ) const
{

  return m_GridLevels[ l ].g_Centering;

}


template < unsigned int VDimension >
typename GridsHierarchy< VDimension >::ImageType::Pointer
GridsHierarchy< VDimension >
::CreateImageAtLevel( const unsigned int l ) const
{

  typename ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions( m_GridLevels[ l ].g_Region );
  outputImage->Allocate();
  outputImage->SetSpacing( m_GridLevels[ l ].g_Spacing );

  return outputImage;

}


template < unsigned int VDimension >
void
GridsHierarchy< VDimension >
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


  // Removing useless offsets from active list in the 3D case
  if ( VDimension == 3 )
  {
    typedef std::list< OffsetType > OffsetListType;
    OffsetListType activeOffsets = grid->g_CoarseOperator->GetActiveOffsetList();
    bool remove;

    for ( typename OffsetListType::iterator activeOffsetsIterator = activeOffsets.begin();
          activeOffsetsIterator != activeOffsets.end(); ++activeOffsetsIterator )
    {

      remove = true;

      for ( unsigned int d = 0; d < VDimension; ++d )
      {
        if ( ( *activeOffsetsIterator )[ d ] == 0 ) remove = false;
      }

      if ( remove ) grid->g_CoarseOperator->DeactivateOffset( *activeOffsetsIterator );

    }
  }


}

} // end namespace mad

} // end namespace itk

#endif  /* __itkGridsHierarchy_hxx */
