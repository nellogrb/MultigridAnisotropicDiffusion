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

#ifndef __itkMultigridAnisotropicDiffusionImageFilter_hxx
#define __itkMultigridAnisotropicDiffusionImageFilter_hxx

#include <cstdlib>
#include <iostream>

#ifdef BENCHMARK
#include <fstream>
#endif

#include "itkImageDuplicator.h"

#include "itkMultigridAnisotropicDiffusionImageFilter.h"

namespace itk
{

template < class TInputImage, class TOutputImage, class TSmootherType >
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::MultigridAnisotropicDiffusionImageFilter() :
  m_TimeStep( 0.01 ),
  m_NumberOfSteps( 1 ),
  m_Cycle( VCYCLE ),
  m_IterationsPerGrid( 2 ),
  m_Tolerance( 1e-6 ),
  m_MaxCycles( 100 ),
  m_Verbose( 0 ),
  m_CurrentLevel( 0 ),
  m_CoarsestLevel( 0 ),
  m_Grids( nullptr ),
  m_DirectSolver( nullptr )
{

}


template < class TInputImage, class TOutputImage, class TSmootherType >
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::~MultigridAnisotropicDiffusionImageFilter()
{

    delete m_Grids;
    delete m_DirectSolver;

};


template < class TInputImage, class TOutputImage, class TSmootherType >
void
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::SetDiffusionTensor( const InputTensorImageType * inputTensor )
{

  // Converting & copying the diffusion tensor to internal pixel type
  m_DiffusionTensor = InternalTensorImageType::New();
  m_DiffusionTensor->SetRegions( inputTensor->GetLargestPossibleRegion() );
  m_DiffusionTensor->Allocate();
  m_DiffusionTensor->SetSpacing( inputTensor->GetSpacing() );

  ImageRegionConstIterator< InputTensorImageType > tensorIterator( inputTensor, inputTensor->GetLargestPossibleRegion() );
  ImageRegionIterator< InternalTensorImageType > memberTensorIterator( m_DiffusionTensor, inputTensor->GetLargestPossibleRegion() );

  unsigned int ImageDimension = TInputImage::ImageDimension;

  while ( !tensorIterator.IsAtEnd() )
    {

    for ( unsigned int d = 0; d < ImageDimension; ++d )
      {
      for ( unsigned int d2 = 0; d2 < d + 1; ++d2 )
        {

        memberTensorIterator.Value()( d, d2 ) = static_cast< InternalPixelType >( tensorIterator.Value()( d, d2 ) );

        }
      }

    ++tensorIterator;
    ++memberTensorIterator;

    }

}


template < class TInputImage, class TOutputImage, class TSmootherType >
void
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::SetInput( const InputImageType * inputImage )
{

  this->ProcessObject::SetNthInput( 0, const_cast< InputImageType * >( inputImage ) );

}


template < class TInputImage, class TOutputImage, class TSmootherType >
void
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::GenerateData()
{

  // Copying and casting input image to internal pixel type
  typename InternalImageType::Pointer rhsImage = InternalImageType::New();
  rhsImage->SetRegions( this->GetInput()->GetLargestPossibleRegion() );
  rhsImage->Allocate();
  rhsImage->SetSpacing( this->GetInput()->GetSpacing() );

  ImageRegionConstIterator< InputImageType > inputIterator( this->GetInput(), this->GetInput()->GetLargestPossibleRegion() );
  ImageRegionIterator< InternalImageType > rhsIterator( rhsImage, this->GetInput()->GetLargestPossibleRegion() );

  while ( !inputIterator.IsAtEnd() )
    {

    rhsIterator.Set( static_cast< InternalPixelType >( inputIterator.Get() ) );

    ++inputIterator;
    ++rhsIterator;

    }


  // Generating grids hierarchy
  m_Grids = new GridsHierarchyType( rhsImage->GetLargestPossibleRegion(), rhsImage->GetSpacing(), m_DiffusionTensor, m_TimeStep );

  m_CoarsestLevel = m_Grids->GetMaxDepth();


  // Generating coarse operators
  //CoarseGridOperatorsGeneratorType coarseOperatorsGenerator( m_CoarseGridOperator );
  //coarseOperatorsGenerator.GenerateOperators( m_Grids, m_DiffusionTensor, m_TimeStep );


  // Creating direct solver on the coarsest level. At the same time, the direct solver
  // generates the decomposition of the operator to be used each time without the need
  // of re-calculating it
  m_DirectSolver = new DirectSolverType( m_Grids->GetCoarseOperatorAtLevel( m_CoarsestLevel ) );


#ifdef BENCHMARK

  m_BenchmarkOutput.open("benchmark.txt");

#endif

  // The real cycle
  typename InternalImageType::Pointer residualImage;
  typename InternalImageType::Pointer solutionImage;
  TSmootherType smoother;

  for ( unsigned int n = 0; n < m_NumberOfSteps; ++n )
    {

#ifdef BENCHMARK

    m_Time = clock();

#endif

    if ( m_NumberOfSteps > 1 &&  m_Verbose )  std::cout << std::endl << "------------ Time step n. " << n + 1
                                                        << " / " << m_NumberOfSteps << "------------" << std::endl;

    if ( m_Cycle == FMG )
      {

      if ( m_Verbose ) std::cout << "|--- Full Multigrid Cycle ---|" << std::endl;
      solutionImage = this->FullMultiGrid( rhsImage );

      }
    else
      {

      // If the cycle type was not set to FMG, we use the solution at the previous
      // time step as initial guess which is hence copied
      solutionImage = InternalImageType::New();

      solutionImage->SetRegions( rhsImage->GetLargestPossibleRegion() );
      solutionImage->Allocate();
      solutionImage->SetSpacing( rhsImage->GetSpacing() );

      rhsIterator.GoToBegin();
      ImageRegionIterator< InternalImageType > initialGuessIterator( solutionImage, solutionImage->GetLargestPossibleRegion() );

      while ( !initialGuessIterator.IsAtEnd() )
        {

        initialGuessIterator.Set( rhsIterator.Get() );

        ++initialGuessIterator;
        ++rhsIterator;

        }

      }

    Precision relativeResidual;
    Precision rhsNorm = this->L2Norm( rhsImage );
    unsigned int numberOfIterations = 0;

    do
      {

      if ( m_Cycle == SMOOTHER )
        {

        solutionImage = smoother.SingleIteration( solutionImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( 0 ) );

        residualImage = smoother.ComputeResidual( solutionImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( 0 ) );

        relativeResidual = this->L2Norm( residualImage ) / rhsNorm;

        if ( m_Verbose ) std::cout << "Smoother iteration n. " << numberOfIterations + 1
                                   << ": relative residual = " << relativeResidual << std::endl;

#ifdef BENCHMARK

        clock_t diff = clock() - m_Time;
        m_BenchmarkOutput << relativeResidual << "_" << ( ( float ) diff ) / CLOCKS_PER_SEC << std::endl;

#endif

        }
      else
        {

        if ( m_Verbose ) std::cout << std::endl << "|--- VCycle n. " << numberOfIterations + 1 << " ---|" << std::endl;

        solutionImage = this->VCycle( solutionImage, rhsImage );

        residualImage = smoother.ComputeResidual( solutionImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( 0 ) );

        relativeResidual = this->L2Norm( residualImage ) / rhsNorm;

        }

      ++numberOfIterations;

      }
    while ( relativeResidual > m_Tolerance && numberOfIterations < m_MaxCycles );

    rhsIterator.GoToBegin();
    ImageRegionIterator< InternalImageType > copyIterator( solutionImage, solutionImage->GetLargestPossibleRegion() );

    // Updating the right hand side of the equation with the solution computed at the
    // current, completed, time step
    while ( !copyIterator.IsAtEnd() )
      {

      rhsIterator.Set( copyIterator.Get() );

      ++copyIterator;
      ++rhsIterator;

      }

    }


  // Casting solution to the desired type and placing it in the output buffer
  typename OutputImageType::Pointer outputImage = OutputImageType::New();

  outputImage->SetRegions( solutionImage->GetLargestPossibleRegion() );
  outputImage->Allocate();
  outputImage->SetSpacing( solutionImage->GetSpacing() );

  ImageRegionConstIterator< InternalImageType > solutionIterator( solutionImage, solutionImage->GetLargestPossibleRegion() );
  ImageRegionIterator< OutputImageType > outputIterator( outputImage, solutionImage->GetLargestPossibleRegion() );

  while ( !solutionIterator.IsAtEnd() )
    {

    outputIterator.Set( static_cast< OutputPixelType >( solutionIterator.Get() ) );

    ++solutionIterator;
    ++outputIterator;

    }

  outputImage->SetOrigin( this->GetInput()->GetOrigin() );

  this->AllocateOutputs();
  this->GraftOutput( outputImage );

#ifdef BENCHMARK

  m_BenchmarkOutput.close();

#endif

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >::InternalImageType::Pointer
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::FullMultiGrid( const InternalImageType * rhsImage )
{

  typename InternalImageType::Pointer outputImage;

  if ( m_CurrentLevel == m_CoarsestLevel )
    {

    outputImage = m_Grids->CreateImageAtLevel( m_CoarsestLevel );
    outputImage->FillBuffer( 0 );

    for ( unsigned int n = 0; n < m_IterationsPerGrid; ++n ) outputImage = this->VCycle( outputImage, rhsImage );

  }
  else {

    InterGridOperatorsType IGOperators( m_Grids->GetVertexCenteringAtLevel( m_CurrentLevel + 1 ) );

    ++m_CurrentLevel;

    typename InternalImageType::Pointer outputImageRestricted;
    typename InternalImageType::Pointer rhsImageRestricted = IGOperators.Restriction( rhsImage );

    outputImageRestricted = this->FullMultiGrid( rhsImageRestricted );

    --m_CurrentLevel;

    outputImage = IGOperators.Interpolation( outputImageRestricted );

    for ( unsigned int n = 0; n < m_IterationsPerGrid; ++n ) outputImage = this->VCycle( outputImage, rhsImage );

  }

  return outputImage;

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >::InternalImageType::Pointer
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::VCycle( const InternalImageType * inputImage, const InternalImageType * rhsImage )
{


  typename InternalImageType::Pointer outputImage;
  typename InternalImageType::Pointer residualImage;

  Precision relativeResidual = 0;
  Precision rhsNorm = this->L2Norm( rhsImage );

  TSmootherType smoother;

  if ( m_CurrentLevel == m_CoarsestLevel )
    {

      outputImage = m_DirectSolver->Solve( rhsImage );
      residualImage = smoother.ComputeResidual( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

      relativeResidual = this->L2Norm( residualImage ) / rhsNorm;
      if ( m_Verbose )
        {

        for ( unsigned int i = 0; i < m_CurrentLevel + 1; ++i ) std::cout << " ";
        std::cout << "Level " << m_CurrentLevel << ", direct solver: relative residual = " << relativeResidual << std::endl;

        }

    }
  else
    {

    typedef ImageDuplicator< InternalImageType > ImageDuplicatorType;
    typename ImageDuplicatorType::Pointer imageDuplicator = ImageDuplicatorType::New();
    imageDuplicator->SetInputImage( inputImage );
    imageDuplicator->Update();
    outputImage = imageDuplicator->GetOutput();


    InterGridOperatorsType IGOperators( m_Grids->GetVertexCenteringAtLevel( m_CurrentLevel + 1 ) );

    for ( unsigned int n = 0; n < m_IterationsPerGrid; ++n )
      {

      outputImage = smoother.SingleIteration( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

      residualImage = smoother.ComputeResidual( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

      relativeResidual = this->L2Norm( residualImage ) / rhsNorm;

      if ( m_Verbose )
        {

        for ( unsigned int i = 0; i < m_CurrentLevel + 1; ++i ) std::cout << " ";
        std::cout << "Level " << m_CurrentLevel << ", iteration " << n + 1 << ": relative residual = " << relativeResidual << std::endl;

        }

#ifdef BENCHMARK

        if ( m_CurrentLevel == 0 )
          {
          clock_t diff = clock() - m_Time;
          m_BenchmarkOutput << relativeResidual << "_" << ( ( float ) diff ) / CLOCKS_PER_SEC << std::endl;
          }

#endif

      }

    typename InternalImageType::Pointer residualCoarse = IGOperators.Restriction( residualImage );

    typename InternalImageType::Pointer solutionCoarse = m_Grids->CreateImageAtLevel( m_CurrentLevel + 1 );
    solutionCoarse->FillBuffer( 0 );

    ++m_CurrentLevel;
    solutionCoarse = this->VCycle( solutionCoarse, residualCoarse );
    --m_CurrentLevel;

    typename InternalImageType::Pointer correctionImage = IGOperators.Interpolation( solutionCoarse );

    ImageRegionIterator< InternalImageType > originalIterator( outputImage, outputImage->GetLargestPossibleRegion() );
    ImageRegionIterator< InternalImageType > correctionIterator( correctionImage, outputImage->GetLargestPossibleRegion() );

    while ( !originalIterator.IsAtEnd() )
      {

      originalIterator.Value() += correctionIterator.Value();

      ++originalIterator;
      ++correctionIterator;

      }

    residualImage = smoother.ComputeResidual( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

    relativeResidual = this->L2Norm( residualImage ) / rhsNorm;

    if ( m_Verbose )
      {

      for ( unsigned int i = 0; i < m_CurrentLevel + 1 ; ++i ) std::cout << " ";

      std::cout << "Level " << m_CurrentLevel << ", initial relative residual = " << relativeResidual << std::endl;

      }

#ifdef BENCHMARK

        if ( m_CurrentLevel == 0 )
          {
          clock_t diff = clock() - m_Time;
          m_BenchmarkOutput << relativeResidual << "_" << ( ( float ) diff ) / CLOCKS_PER_SEC << std::endl;
          }

#endif

    for ( unsigned int n = 0; n < m_IterationsPerGrid; ++n )
      {

      outputImage = smoother.SingleIteration( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

      residualImage = smoother.ComputeResidual( outputImage, rhsImage, m_Grids->GetCoarseOperatorAtLevel( m_CurrentLevel ) );

      relativeResidual = this->L2Norm( residualImage ) / rhsNorm;

      if ( m_Verbose )
        {

        for ( unsigned int i = 0; i < m_CurrentLevel + 1; ++i ) std::cout << " ";
        std::cout << "Level " << m_CurrentLevel << ", iteration " << n + 1 << ": relative residual = " << relativeResidual << std::endl;

        }

#ifdef BENCHMARK

        if ( m_CurrentLevel == 0 )
          {
          clock_t diff = clock() - m_Time;
          m_BenchmarkOutput << relativeResidual << "_" << ( ( float ) diff ) / CLOCKS_PER_SEC << std::endl;
          }

#endif

      }

    }

  return outputImage;

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >::Precision
MultigridAnisotropicDiffusionImageFilter< TInputImage, TOutputImage, TSmootherType >
::L2Norm( const InternalImageType * inputImage ) const
{

  Precision norm = 0;
  ImageRegionConstIterator< InternalImageType > inputIterator( inputImage, inputImage->GetLargestPossibleRegion() );

  while ( !inputIterator.IsAtEnd() )
    {

    norm += inputIterator.Get() * inputIterator.Get();
    ++inputIterator;

    }

  return sqrt( norm );

}



} // end namespace itk

#endif  /* __itkMultigridAnisotropicDiffusionImageFilter_hxx */
