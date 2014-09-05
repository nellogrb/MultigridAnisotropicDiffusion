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
#ifndef __itkInterGridOperators_h
#define __itkInterGridOperators_h

#include <array>
#include <map>

#include "itkNeighborhood.h"
#include "itkImage.h"

namespace itk
{

namespace mad
{

/** \class InterGridOperators
 *
 * \brief Class implementation of the two inter grid operators. It has to
 * be initialized with an array prescribing which kind of coarsening has
 * to be used between the two grids on each direction: vertex centered vs
 * cell centered. It does not need other informations on the grids, as they
 * are extracted from the input image when Interpolation or Restriction are called.
 *
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class ITK_EXPORT InterGridOperators
{
public:

  /** Standard class typedefs. */
  typedef double                                                                 Precision;
  typedef InterGridOperators                                                     Self;
  typedef Image< Precision, VDimension >                                         ImageType;
  typedef Neighborhood< Precision, VDimension >                                  StencilType;

  typedef ImageRegion< VDimension >                                              ImageRegionType;
  typedef typename ImageType::SpacingType                                        SpacingType;
  typedef typename ImageType::SizeType                                           SizeType;
  typedef typename ImageType::IndexType                                          IndexType;
  typedef typename ImageType::OffsetType                                         OffsetType;

  /** Characterization of a point based on its position relative to the interior
   *  of the image region. */
  enum PointPositionType { left, interior, right };

  /** Class constructor which takes as argument whether the coarse grid is obtained
   * via vertex centered versus cell centered approach. */
  InterGridOperators( const std::array< bool, VDimension > & vertexCentering );

  /** Class destructor. */
  ~InterGridOperators() {};

  /** Interpolation operator. It optionally accepts an array whose
   *  boolean values prescribe if the border points, on the left and right
   *  side of each direction, has to be restricted with the same stencil
   *  as that of interior points. */
  typename ImageType::Pointer Interpolation( const ImageType * inputImage,
                                             const bool * ignoreLeftBorder = nullptr,
                                             const bool * ignoreRightBorder = nullptr ) const;

  /** Restriction operator. It optionally accepts an array whose
   *  boolean values prescribe if the border points, on the left and right
   *  side of each direction, has to be restricted with the same stencil
   *  as that of interior points. */
  typename ImageType::Pointer Restriction( const ImageType * inputImage,
                                           const bool * ignoreLeftBorder = nullptr,
                                           const bool * ignoreRightBorder = nullptr ) const;


private:

  /** Approach used in each direction. */
  std::array< bool, VDimension > m_VertexCentering;

  /** Utility function to generate stencils used by both interpolation and restriction.
   *  This is achieved by composition of the 1-dimensional stencils vertexCentered1D and
   *  cellCentered1D. pointPosition determines the position of the central pixel with respect
   *  to the interior of the image region. */
  StencilType GenerateStencil( const std::array< PointPositionType, VDimension > & pointPosition,
                               const std::map< PointPositionType, std::array< Precision, 3 > > & vertexStencils1D,
                               const std::map< PointPositionType, std::array< Precision, 5 > > & cellStencils1D ) const;


  /** Monodimensional interpolation and restriction stencils for both vertex centered and
   *  cell centered coarsening. */
  const std::map< PointPositionType, std::array< Precision, 3 > > interpolationVertexStencils1D =
    {
      { left, {{ 0. , 1. , 1. / 2. }} },
      { interior, {{ 1. / 2. , 1. , 1. / 2. }} },
      { right, {{ 1. / 2. , 1. , 0. }} }
    };

  const std::map< PointPositionType, std::array< Precision, 5 > > interpolationCellStencils1D =
    {
      { left, {{ 0.,  0. , 1. , 3. / 4. , 1. / 4. }} },
      { interior, {{ 0. , 1. / 4. , 3. / 4. , 3. / 4. , 1. / 4. }} },
      { right, {{ 0. , 1. / 4. , 3. / 4. , 1. , 0. }} }
    };

  const std::map< PointPositionType, std::array< Precision, 3 > > restrictionVertexStencils1D =
    {
      { left, {{ 0. , 1. , 0. }} },
      { interior, {{ 1. / 4. , 1. / 2. , 1. / 4. }} },
      { right, {{ 0. , 1. , 0. }} }
    };

  const std::map< PointPositionType, std::array< Precision, 5 > > restrictionCellStencils1D =
    {
      { left, {{ 0. , 0. , 1. / 2. , 3. / 8. , 1. / 8. }} },
      { interior, {{ 0. , 1. / 8. , 3. / 8. , 3. / 8. , 1. / 8. }} },
      { right, {{ 0. , 1. / 8. , 3. / 8. , 1. / 2. , 0. }} }
    };


  /** Purposely not implemented */
  InterGridOperators( const Self & );
  void operator=( const Self & );


};


} // end namespace mad

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkInterGridOperators.hxx"
#endif

#endif /* __itkInterGridOperators_h */
