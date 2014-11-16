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
#ifndef __itkStencilImage_h
#define __itkStencilImage_h

#include "itkNeighborhood.h"
#include "itkImage.h"
#include <list>

namespace itk
{


namespace mad
{

/** \class StencilImage
 *
 * \brief This class is derived from a specialization of the Image class, where the
 * PixelType is defined as a Neighborhood of values: it is however templated in
 * order to mimic an Image class.
 * It is used by the other classes in the ITKMultigridAnisotropicDiffusion, and
 * can been interpreted as something similar to a sparse matrix where every row
 * corresponds to a single pixel (i.e., a Neighborhood).
 * It also expands the base class, containing a list of active offsets; this
 * is useful in order to easily get the neighbors which need to be visited.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */
template < class TPixelType, unsigned int VDimension >
class StencilImage : public Image< Neighborhood< TPixelType, VDimension >, VDimension >
{
public:

  /** Standard class typedefs. */
  typedef StencilImage                                                      Self;
  typedef Image< Neighborhood< TPixelType, VDimension>, VDimension >        SuperClass;
  typedef SmartPointer< Self >                                              Pointer;
  typedef SmartPointer< const Self >                                        ConstPointer;

  typedef Neighborhood< TPixelType, VDimension>                             StencilType;

  typedef Image< TPixelType, VDimension >                                   ImageType;
  typedef typename ImageType::OffsetType                                    OffsetType;
  typedef std::list< OffsetType >                                           OffsetListType;
  typedef typename ImageType::SizeType                                      SizeType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StencilImage, Image );

  /** Actives all offsets inside the radius. */
  void ActivateAllOffsets ();

  /** Gets the list of active offsets. */
  itkGetConstMacro( ActiveOffsetList, OffsetListType );

  /** Gets the number of active offsets. */
  typename OffsetListType::size_type GetActiveOffsetListSize () const;

  /** Gets the radius of the neighborhood. */
  itkGetConstMacro( Radius, SizeType );

  /** Sets the radius of the neighborhood. */
  itkSetMacro( Radius, SizeType );


protected:

  /** Class constructor. */
  StencilImage();

  /** Class destructor. */
  virtual ~StencilImage() {};

private:

  SizeType                                    m_Radius;
  OffsetListType                              m_ActiveOffsetList;

  /** Purposely not implemented */
  StencilImage( const Self & );
  Self & operator= ( const Self & );

};

} // end namespace mad

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStencilImage.hxx"
#endif

#endif /* __itkStencilImage_h */
