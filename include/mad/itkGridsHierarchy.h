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
#ifndef __itkGridsHierarchy_h
#define __itkGridsHierarchy_h

#include <array>

#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkSymmetricSecondRankTensor.h"

#include "itkStencilImage.h"
#include "itkInterGridOperators.h"


namespace itk
{

namespace mad
{

/** \class GridsHierarchy
 *
 * \brief Class which creates and holds a hierarchy of grids to be used
 * by itkMultigridAnisotropicDiffusion. It contains informations about
 * each level's grid: spacing, region, coarse operator in the form of
 * a pointer to a StencilImage, and whether the level was obtained
 * with a vertex centered or a cell centered coarsening (as a convention,
 * level 0 has a vertex centered approach in each direction).
 *
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class GridsHierarchy
{
public:

  /** Standard class typedefs. */
  typedef double                                                                    Precision;
  typedef GridsHierarchy                                                            Self;

  typedef Image< Precision, VDimension >                                            ImageType;
  typedef Neighborhood< Precision, VDimension >                                     StencilType;
  typedef StencilImage< Precision, VDimension >                                     StencilImageType;
  typedef SymmetricSecondRankTensor< Precision, VDimension >                        TensorType;
  typedef Image< TensorType, VDimension >                                           TensorImageType;
  typedef mad::InterGridOperators< VDimension >                                     InterGridOperatorsType;

  typedef ImageRegion< VDimension >                                                 ImageRegionType;
  typedef typename ImageType::SpacingType                                           SpacingType;
  typedef typename ImageType::SizeType                                              SizeType;
  typedef typename ImageType::IndexType                                             IndexType;
  typedef typename ImageType::OffsetType                                            OffsetType;

  typedef typename InterGridOperators< VDimension >::CoarseGridCenteringType        CoarseGridCenteringType;

  /** Raw data structure for each level */
  struct Grid {

    ImageRegionType                                             g_Region;
    SpacingType                                                 g_Spacing;
    typename StencilImageType::Pointer                          g_CoarseOperator;
    std::array< CoarseGridCenteringType, VDimension >           g_Centering;

  };

  /** Returns pointer to the coarse operator at level l. */
  typename StencilImageType::Pointer GetCoarseOperatorAtLevel( const unsigned int l ) const;

  /** Returns pointer to the grid raw data at level l. */
  Grid * GetGridAtLevel( const unsigned int l );

  /** Creates an empty (allocated but not initialized) defined on level l,
   *  and returns a pointer to it. */
  typename ImageType::Pointer CreateImageAtLevel( const unsigned int l ) const;

  /** Returns the maximum depth of the hierarchy. */
  unsigned int GetMaxDepth() const;

  /** Returns the region at level l. */
  ImageRegionType GetRegionAtLevel( const unsigned int l ) const;

  /** Returns the spacing at level l. */
  SpacingType GetSpacingAtLevel( const unsigned int l ) const;

  /** Returns pointer to the coarse operator at level l. */
  std::array< CoarseGridCenteringType, VDimension > GetVertexCenteringAtLevel( const unsigned int l ) const;

  /** Class constructor. It needs the grid region and the spacing at level 0. */
  GridsHierarchy( const ImageRegionType & initialRegion, const SpacingType & initialSpacing,
                  const TensorImageType * fineTensor, const Precision timeStep );

  /** Class destructor. */
  ~GridsHierarchy();

private:

  Grid *                           m_GridLevels;
  unsigned int                     m_MaxDepth;

  /** Direct Coarse Approximation implementation. */
  void GenerateDCA( Grid * grid,
                    const TensorImageType * tensor,
                    const Precision timeStep ) const;

  /** Purposely not implemented */
  GridsHierarchy( const Self & );
  Self & operator= ( const Self & );

};


} // end namespace mad

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGridsHierarchy.hxx"
#endif

#endif /* __itkGridsHierarchy_h */
