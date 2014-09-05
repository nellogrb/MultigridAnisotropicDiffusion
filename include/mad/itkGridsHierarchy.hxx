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
::GridsHierarchy( const ImageRegionType & initialRegion, const SpacingType & initialSpacing )
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
  m_GridLevels[ 0 ].g_VertexCentered.fill( 1 );

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
              m_GridLevels[ l ].g_VertexCentered[ d ] = false;

            }
          else
            {

              coarseRegionSize[ d ] = ( m_GridLevels[ l - 1 ].g_Region.GetSize( d ) - 1 ) / 2 + 1;
              m_GridLevels[ l ].g_VertexCentered[ d ] = true;

            }

        }

      ImageRegionType coarseRegion( coarseRegionOrigin, coarseRegionSize );
      m_GridLevels[ l ].g_Region = coarseRegion;

      m_GridLevels[ l ].g_CoarseOperator = 0;

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
std::array< bool, VDimension >
GridsHierarchy< VDimension >
::GetVertexCenteringAtLevel( const unsigned int l ) const
{

  return m_GridLevels[ l ].g_VertexCentered;

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

} // end namespace mad

} // end namespace itk

#endif  /* __itkGridsHierarchy_hxx */
