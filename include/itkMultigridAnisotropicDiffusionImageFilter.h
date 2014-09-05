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
#ifndef __itkMultigridAnisotropicDiffusionImageFilter_h
#define __itkMultigridAnisotropicDiffusionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkMacro.h"

#include "mad/itkStencilImage.h"
#include "mad/itkGridsHierarchy.h"
#include "mad/itkInterGridOperators.h"
#include "mad/itkCoarseGridOperatorsGenerator.h"
#include "mad/itkMultigridGaussSeidelSmoother.h"
#include "mad/itkDirectSolver.h"

#ifdef BENCHMARK
#include <fstream>
#include <ctime>
#endif

namespace itk
{

/** \class MultigridAnisotropicDiffusionImageFilter
 *
 * \brief This class embodies, in the form of an ImageToImageFilter class,
 * an implementation of a multigrid method to solve a generic anisotropic
 * diffusion problem:
 *
 * \f[
 *   \partial_t I(\mathbf{x}, t) - \mathrm{div} \left( M(\mathbf{x}) \nabla I(\mathbf{x}, t) \right) = 0
 * \f]
 *
 * The filter is templated, in addition to the input and output image
 * types, over the smoother type; the latter should be a derived class
 * of the pure virtual base class MultigridSmoother. The smoothers
 * MultigridGaussSeidel (which is the default one if not specified)
 * and MultigridWeightedJacobi have already been implemented, and can
 * be found in the subdirectory .mad/ .
 *
 * The minimum inputs required for the filter to work are:
 *
 *   -# The image to be diffused
 *   -# A diffusion tensor in the form of an image with a
 *   SymmetricSecondRankTensor as pixel type
 *
 * The image and the tensor must be defined on the same ImageRegion.
 * Optional parameters are:
 *
 *   -# The time step (defaults to 0.01)
 *   -# The number of time steps (defaults to 1)
 *   -# The type of coarse grid operator to build, the options being
 *   Direct Coarse Approximation (DCA) and Galerkin Coarse Approximation
 *   (GCA) (defaults to DCA)
 *   -# The type of cycle to be executed: FMG, VCYCLE and SMOOTHER
 *   (in the last two cases, the initial guess is the image at the
 *   previous step). FMG and VCYCLE are well-known multigrid cycles.
 *   The SMOOTHER mode solves the problem using the chosen smoother
 *   only, and is mainly intended as a baseline for comparison with
 *   the first two. (defaults to VCYCLE).
 *   -# The number of smoother iterations for each level (defaults to 2)
 *   -# The maximum relative residual tolerance (defaults to 1e-6)
 *   -# The maximum number of VCycles (defaults to 100)
 *   -# The verbosity (defaults to 0 = quiet)
 *
 * The filter works with double type as internal precision. The coefficients
 * of the SymmetricSecondRankTensor elements composing the diffusion
 * tensor are expected to have the same type as the input image.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */
template < class TInputImage, class TOutputImage,
           class TSmootherType = mad::MultigridGaussSeidelSmoother< TInputImage::ImageDimension > >
class ITK_EXPORT MultigridAnisotropicDiffusionImageFilter :
  public ImageToImageFilter< TInputImage, TOutputImage >
{
public:

  /** Standard and useful class typedefs. */
  typedef MultigridAnisotropicDiffusionImageFilter                                                   Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage >                                            SuperClass;
  typedef SmartPointer< Self >                                                                       Pointer;
  typedef SmartPointer< const Self >                                                                 ConstPointer;

  typedef TInputImage                                                                                InputImageType;
  typedef typename TInputImage::PixelType                                                            InputPixelType;
  typedef TOutputImage                                                                               OutputImageType;
  typedef typename TOutputImage::PixelType                                                           OutputPixelType;


  typedef double                                                                                     InternalPixelType;
  typedef Image< InternalPixelType, TInputImage::ImageDimension >                                    InternalImageType;

  typedef Image< SymmetricSecondRankTensor< InputPixelType, TInputImage::ImageDimension >,
                 TInputImage::ImageDimension >                                                       InputTensorImageType;
  typedef Image< SymmetricSecondRankTensor< InternalPixelType, TInputImage::ImageDimension >,
                 TInputImage::ImageDimension >                                                       InternalTensorImageType;

  typedef mad::StencilImage< InternalPixelType, TInputImage::ImageDimension >                        StencilImageType;
  typedef typename mad::GridsHierarchy< TInputImage::ImageDimension >                                GridsHierarchyType;
  typedef typename mad::InterGridOperators< TInputImage::ImageDimension >                            InterGridOperatorsType;
  typedef typename mad::CoarseGridOperatorsGenerator< TInputImage::ImageDimension >                  CoarseGridOperatorsGeneratorType;
  typedef typename mad::CoarseGridOperatorsGenerator< TInputImage::ImageDimension >
                      ::CoarseGridOperatorType                                                       CoarseGridOperatorType;
  typedef typename mad::DirectSolver< TInputImage::ImageDimension >                                  DirectSolverType;

  typedef InternalPixelType                                                                          Precision;

  enum CycleType { VCYCLE, FMG, SMOOTHER };

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultigridAnisotropicDiffusionImageFilter, ImageToImageFilter );

  /** Sets the type of coarse grid operator to be built by the
   * class CoarseGridOperatorsGenerator. CoarseGridOperatorType is
   * an enum with DCA and GCA as possible values. */
  itkSetMacro( CoarseGridOperator, CoarseGridOperatorType );

  /** Sets the type of cycle to be executed. CycleType is
   *  an enum with VCYCLE, FMG and SMOOTHER as possible values */
  itkSetMacro( Cycle, CycleType );

  /** Sets the number of iterations that should be executed each time
   *  on both the ascending and descending legs of the VCycle. */
  itkSetMacro( IterationsPerGrid, unsigned int );

  /** Sets the maximum number of VCycle to be executed for each
   *  time step if the residual tolerance is not reached before. */
  itkSetMacro( MaxCycles, unsigned int );

  /** Sets the number of time steps. All of the data required
   *  to solve the problem is calculated just once, before the
   *  first step. */
  itkSetMacro( NumberOfSteps, unsigned int );

  /** Sets the time step. */
  itkSetMacro( TimeStep, Precision );

  /** Sets the tolerance for the residual, for each time step. */
  itkSetMacro( Tolerance, Precision );

  /** Sets wether the filter should produce textual output containing
   *  informations on the current status. */
  itkSetMacro( Verbose, bool );

  /** Sets the diffusion tensor, whose elements are internally casted to
   *  InternalPixelPrecision. */
  void SetDiffusionTensor( const InputTensorImageType * inputTensor );

  /** Sets the input image, whose elements are internally casted to
   *  InternalPixelPrecision. */
  void SetInput( const InputImageType * inputImage );

protected:

  /** Class constructor. */
  MultigridAnisotropicDiffusionImageFilter();

  /** Class destructor. */
  ~MultigridAnisotropicDiffusionImageFilter();

  /** Generates the output, which is then accessed by method GetOuput(). */
  virtual void GenerateData();

private:

  typename InternalTensorImageType::Pointer         m_DiffusionTensor;
  Precision                                         m_TimeStep;

  unsigned int                                      m_NumberOfSteps;
  CoarseGridOperatorType                            m_CoarseGridOperator;
  CycleType                                         m_Cycle;
  unsigned int                                      m_IterationsPerGrid;
  Precision                                         m_Tolerance;
  unsigned int                                      m_MaxCycles;
  bool                                              m_Verbose;

  unsigned int                                      m_CurrentLevel;
  unsigned int                                      m_CoarsestLevel;

  GridsHierarchyType *                              m_Grids;
  DirectSolverType *                                m_DirectSolver;

#ifdef BENCHMARK
  std::ofstream                                     m_BenchmarkOutput;
  clock_t                                           m_Time;
#endif

  /** Returns the approximated solution after a Full Multigrid Cycle. */
  typename InternalImageType::Pointer FullMultiGrid( const InternalImageType * rhsImage );

  /** Returns the L2-Norm of inputImage. */
  Precision L2Norm( const InternalImageType * inputImage ) const;

  /** Returns the approximated solution after a single VCycle. */
  typename InternalImageType::Pointer VCycle( const InternalImageType * inputImage,
                                              const InternalImageType * rhsImage );

  /** Purposely not implemented */
  MultigridAnisotropicDiffusionImageFilter( const Self & );
  void operator=( const Self & );

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultigridAnisotropicDiffusionImageFilter.hxx"
#endif

#endif // __itkMultigridAnisotropicDiffusionImageFilter_h
