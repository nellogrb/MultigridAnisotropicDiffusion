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

#ifndef __itkStencilImage_hxx
#define __itkStencilImage_hxx

#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

template < class TPixelType, unsigned int VDimension >
StencilImage< TPixelType, VDimension >
::StencilImage()
{

  m_Radius.Fill( 0 );

}


template < class TPixelType, unsigned int VDimension >
typename StencilImage< TPixelType, VDimension >::OffsetListType::size_type
StencilImage< TPixelType, VDimension >
::GetActiveOffsetListSize () const
{

  return m_ActiveOffsetList.size();

}


template < class TPixelType, unsigned int VDimension >
void
StencilImage< TPixelType, VDimension >
::ActivateAllOffsets ()
{

  StencilType n;
  n.SetRadius( m_Radius );

  for ( typename StencilType::NeighborIndexType nIndex = 0; nIndex < n.Size(); ++nIndex )
   {

    m_ActiveOffsetList.push_back( n.GetOffset( nIndex ) );

   }

}


template < class TPixelType, unsigned int VDimension >
void
StencilImage< TPixelType, VDimension >
::ActivateOffset ( const OffsetType & offset )
{

  typename OffsetListType::iterator offsetListIterator;
  offsetListIterator = std::find ( m_ActiveOffsetList.begin(), m_ActiveOffsetList.end(), offset );
  if ( offsetListIterator == m_ActiveOffsetList.end() ) m_ActiveOffsetList.push_back( offset );

}


template < class TPixelType, unsigned int VDimension >
void
StencilImage< TPixelType, VDimension >
::DeactivateOffset ( const OffsetType & offset )
{

  m_ActiveOffsetList.remove( offset );

}


} // end namespace mad

} // end namespace itk

#endif  /* __itkStencilImage_hxx */
