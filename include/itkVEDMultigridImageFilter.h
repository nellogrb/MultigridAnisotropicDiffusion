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
#ifndef __itkVEDMultigridImageFilter_h
#define __itkVEDMultigridImageFilter_h

#include <vector>

#include "itkImageToImageFilter.h"
#include "itkMacro.h"

#include "itkMultigridAnisotropicDiffusionImageFilter.h"
#include "mad/itkMultigridGaussSeidelSmoother.h"

namespace itk
{

/** \class VEDMultigridImageFilter
 *
 * \brief Implementation of the VED method by Manniesing, which internally uses
 * MultigridAnisotropicDiffusionImageFilter for the diffusion steps.
 *
 * Reference for the algorithm:
 * R Manniesing, MA Viergever, WJ Niessen, <em>Vessel enhancing diffusion: A scale space representation of vessel structures</em>,
 * Medical Image Analysis 10(6), 815-825.
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */
template < class TInputImage, class TOutputImage,
           class TSmootherType = mad::MultigridGaussSeidelSmoother< TInputImage::ImageDimension > >
class ITK_EXPORT VEDMultigridImageFilter :
  public ImageToImageFilter< Image< typename TInputImage::PixelType, 3 >, Image< typename TOutputImage::PixelType, 3 > >
{
public:

  /** Standard and useful class typedefs. */
  typedef VEDMultigridImageFilter                                                     Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage >                             SuperClass;
  typedef SmartPointer< Self >                                                        Pointer;
  typedef SmartPointer< const Self >                                                  ConstPointer;

  typedef TInputImage                                                                 InputImageType;
  typedef typename TInputImage::PixelType                                             InputPixelType;
  typedef TOutputImage                                                                OutputImageType;
  typedef typename TOutputImage::PixelType                                            OutputPixelType;

  typedef double                                                                      InternalPixelType;
  typedef Image< InternalPixelType, 3 >                                               InternalImageType;

  typedef Image< SymmetricSecondRankTensor< InternalPixelType, 3 >, 3 >               TensorImageType;

  typedef InternalPixelType                                                           Precision;
  typedef ImageRegion< 3 >                                                            ImageRegionType;

  typedef FixedArray< Precision, 3 >                                                  VectorType;
  typedef Image< VectorType, 3 >                                                      VectorImageType;
  typedef Matrix< Precision, 3, 3 >                                                   MatrixType;
  typedef Image< MatrixType, 3 >                                                      MatrixImageType;

  typedef MultigridAnisotropicDiffusionImageFilter< InternalImageType,
                                                    InternalImageType,
                                                    TSmootherType >                   MADFilterType;

  typedef typename MADFilterType::CycleType                                           CycleType;


  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( VEDMultigridImageFilter, ImageToImageFilter );

  /** Setters for VED parameters. */
  itkSetMacro( Alpha, Precision );
  itkSetMacro( Beta, Precision );
  itkSetMacro( Gamma, Precision );
  itkSetMacro( Epsilon, Precision );
  itkSetMacro( Omega, Precision );
  itkSetMacro( Sensitivity, Precision );
  itkSetMacro( Scales, std::vector< Precision > );
  itkSetMacro( Iterations, unsigned int );
  itkSetMacro( DiffusionIterations, unsigned int );

  /** Setters for MAD parameters. */
  itkSetMacro( Cycle, CycleType );
  itkSetMacro( TimeStep, Precision );
  itkSetMacro( Tolerance, Precision );
  itkSetMacro( DiffusionIterationsPerGrid, unsigned int );

  /** Sets wether the filter should produce textual output containing
   *  informations on the current status. */
  itkSetMacro( Verbose, bool );

protected:

  /** Class constructor. */
  VEDMultigridImageFilter();

  /** Class destructor. */
  ~VEDMultigridImageFilter();

  /** Generates the output. */
  virtual void GenerateData();

private:

  Precision                                      m_Alpha;
  Precision                                      m_Beta;
  Precision                                      m_Gamma;
  Precision                                      m_Epsilon;
  Precision                                      m_Omega;
  Precision                                      m_Sensitivity;
  std::vector< Precision >                       m_Scales;
  unsigned int                                   m_Iterations;
  unsigned int                                   m_DiffusionIterations;

  CycleType                                      m_Cycle;
  Precision                                      m_TimeStep;
  Precision                                      m_Tolerance;
  unsigned int                                   m_DiffusionIterationsPerGrid;

  unsigned int                                   m_CurrentIteration;

  typename InternalImageType::Pointer            m_MaxVesselnessResponse;
  typename TensorImageType::Pointer              m_MaxVesselnessHessian;
  typename VectorImageType::Pointer              m_MaxVesselnessEigenValues;
  typename MatrixImageType::Pointer              m_MaxVesselnessEigenVectors;

  typename TensorImageType::Pointer              m_DiffusionTensor;

  bool                                           m_Verbose;

  /** Returns the Hessian matrix computed in each point of the region */
  typename TensorImageType::Pointer ComputeHessian( const InternalImageType * inputImage, const Precision sigma ) const;

  /** Updates the vesselness measure (and the corresponding eigensystem) at every point,
   *  where inputTensor represents the Hessian matrix */
  void UpdateVesselness( const TensorImageType * inputTensor );

  /** Returns the value of vesselness measure */
  inline Precision VesselnessFunction( const vnl_vector< Precision > eigenvalues ) const;

  /** Generates the diffusion tensor to be applied, using the information
   *  gathered at previous steps */
  void GenerateDiffusionTensor();

  /** Executes m_DiffusionIterations diffusion steps using MultigridAnisotropicImageFilter */
  typename InternalImageType::Pointer DiffusionStep( const InternalImageType * inputImage ) const;

  /** Purposely not implemented */
  VEDMultigridImageFilter( const Self & ); // Purposely not implemented
  void operator=( const Self & ); // Purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVEDMultigridImageFilter.hxx"
#endif

#endif /* __itkVEDMultigridImageFilter_h */
