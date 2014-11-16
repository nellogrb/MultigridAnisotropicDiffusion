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

#ifndef __itkVEDMultigridImageFilter_hxx
#define __itkVEDMultigridImageFilter_hxx

#include "itkHessianRecursiveGaussianImageFilter.h"

#include <vnl/vnl_vector.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>

#include "itkVEDMultigridImageFilter.h"

namespace itk
{

template < class TInputImage, class TOutputImage, class TSmootherType >
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::VEDMultigridImageFilter() :
  // Some default values for the low-dose case used in the original paper
  m_Alpha( 0.5 ),
  m_Beta( 0.5 ),
  m_Gamma( 5.0 ),
  m_Epsilon( 0.01 ),
  m_Omega( 5. ),
  m_Sensitivity( 10. ),
  m_Iterations( 1 ),
  m_DiffusionIterations( 5 ),
  m_Cycle( MADFilterType::CycleType::VCYCLE ),
  m_TimeStep( 0.1 ),
  m_Tolerance( 1e-6 ),
  m_DiffusionIterationsPerGrid( 2 ),
  m_CurrentIteration( 0 ),
  m_Verbose( false )
{

  m_Scales.resize( 5 );

  m_Scales[0] = 0.300;
  m_Scales[1] = 0.482;
  m_Scales[2] = 0.775;
  m_Scales[3] = 1.245;
  m_Scales[4] = 2.000;

}


template < class TInputImage, class TOutputImage, class TSmootherType >
void
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::GenerateData()
{

  ImageRegionType imageRegion = this->GetInput()->GetLargestPossibleRegion();

  // Casting input image to internal pixel type
  typename InternalImageType::Pointer internalInputImage = InternalImageType::New();
  typename InternalImageType::Pointer copyInputImage = InternalImageType::New();

  internalInputImage->SetRegions( imageRegion );
  internalInputImage->Allocate();
  internalInputImage->SetSpacing( this->GetInput()->GetSpacing() );
  internalInputImage->SetOrigin( this->GetInput()->GetOrigin() );

  copyInputImage->SetRegions( imageRegion );
  copyInputImage->Allocate();
  copyInputImage->SetSpacing( this->GetInput()->GetSpacing() );
  copyInputImage->SetOrigin( this->GetInput()->GetOrigin() );

  ImageRegionConstIterator< InputImageType > inputIterator( this->GetInput(), imageRegion );
  ImageRegionIterator< InternalImageType > internalInputIterator( internalInputImage, imageRegion );
  ImageRegionIterator< InternalImageType > copyInputIterator( copyInputImage, imageRegion );

  while ( !inputIterator.IsAtEnd() )
    {

    internalInputIterator.Set( static_cast< InternalPixelType >( inputIterator.Get() ) );
    copyInputIterator.Set( static_cast< InternalPixelType >( inputIterator.Get() ) );

    ++inputIterator;
    ++internalInputIterator;
    ++copyInputIterator;

    }


  typename TensorImageType::Pointer hessianImage;

  for ( m_CurrentIteration = 0; m_CurrentIteration < m_Iterations; ++m_CurrentIteration )
    {

    std::cout << "Iteration n." << m_CurrentIteration + 1 << "..." << std::endl;

    for ( unsigned int i = 0; i < m_Scales.size(); ++i )
      {

      std::cout << "Computing Hessian..." << std::endl;
      hessianImage = this->ComputeHessian( internalInputImage, m_Scales[ i ] );
      std::cout << "Updating vesselness..." << std::endl;
      this->UpdateVesselness( hessianImage );

      }

     std::cout << "Generating tensor..." << std::endl;
     this->GenerateDiffusionTensor();
     m_MaxVesselnessResponse = 0;
     m_MaxVesselnessEigenValues = 0;
     m_MaxVesselnessEigenVectors = 0;

     std::cout << "Applying anisotropic diffusion..." << std::endl;
     internalInputImage = this->DiffusionStep( internalInputImage );

    }


  // Casting solution to the desired type and placing it in the output buffer
  typename OutputImageType::Pointer outputImage = OutputImageType::New();

  outputImage->SetRegions( imageRegion );
  outputImage->Allocate();
  outputImage->SetSpacing( internalInputImage->GetSpacing() );
  outputImage->SetOrigin( internalInputImage->GetOrigin() );

  ImageRegionConstIterator< InternalImageType > solutionIterator( internalInputImage, imageRegion );
  ImageRegionIterator< OutputImageType > outputIterator( outputImage, imageRegion );

  while ( !solutionIterator.IsAtEnd() )
    {

    outputIterator.Set( static_cast< OutputPixelType >( solutionIterator.Get() ) );

    ++solutionIterator;
    ++outputIterator;

    }

  this->AllocateOutputs();
  this->GraftOutput( outputImage );

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >::TensorImageType::Pointer
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::ComputeHessian( const InternalImageType * inputImage, const Precision sigma ) const
{

  typedef HessianRecursiveGaussianImageFilter< InternalImageType, TensorImageType > HessianType;
  typename HessianType::Pointer hessian = HessianType::New();
  hessian->SetInput( inputImage );
  hessian->SetNormalizeAcrossScale( true );
  hessian->SetSigma( sigma );
  hessian->Update();

  return hessian->GetOutput();

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >::Precision
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::VesselnessFunction( const vnl_vector< Precision > eigenvalues ) const
{
  Precision vesselness;

  if ( ( ( eigenvalues[ 1 ] >= 0 ) || ( eigenvalues[ 2 ] >= 0 ) ) )
  {
    vesselness = 0.;
  }
  else
  {

    const Precision smoothC = 1e-5;

    const Precision alphaDenominator = 2.0 * m_Alpha * m_Alpha;
    const Precision betaDenominator = 2.0 * m_Beta * m_Beta;
    const Precision gammaDenominator= 2.0 * m_Gamma * m_Gamma;

    const Precision alphaNumerator = ( eigenvalues[ 1 ] * eigenvalues[ 1 ] ) / ( eigenvalues[ 2 ] * eigenvalues[ 2 ] );
    const Precision betaNumerator = ( eigenvalues[ 0 ] * eigenvalues[ 0 ] ) / abs( eigenvalues[ 1 ] * eigenvalues[ 2 ] );
    const Precision gammaNumerator = ( eigenvalues[ 0 ] * eigenvalues[ 0 ] )
                                   + ( eigenvalues[ 1 ] * eigenvalues[ 1 ] )
                                   + ( eigenvalues[ 2 ] * eigenvalues[ 2 ] );

    const Precision smoothFactor = exp( - ( 2 * smoothC * smoothC ) / ( abs( eigenvalues[ 1 ] )
                                   * eigenvalues[ 2 ] * eigenvalues[ 2 ] ) );

    vesselness = smoothFactor * ( 1. - exp( - alphaNumerator / alphaDenominator ) )
                              * exp( - betaNumerator / betaDenominator )
                              * ( 1. - exp( - gammaNumerator / gammaDenominator ) );
  }

  return vesselness;

}


template < class TInputImage, class TOutputImage, class TSmootherType >
void
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::UpdateVesselness( const TensorImageType * inputTensor )
{

  bool firstTime = !m_MaxVesselnessResponse;

  ImageRegionType inputRegion = inputTensor->GetLargestPossibleRegion();

  if ( firstTime ) {

    m_MaxVesselnessResponse = InternalImageType::New();
    m_MaxVesselnessResponse->SetRegions( inputRegion );
    m_MaxVesselnessResponse->Allocate();
    m_MaxVesselnessResponse->FillBuffer( 0. );

    m_MaxVesselnessEigenValues = VectorImageType::New();
    m_MaxVesselnessEigenValues->SetRegions( inputRegion );
    m_MaxVesselnessEigenValues->Allocate();

    m_MaxVesselnessEigenVectors = MatrixImageType::New();
    m_MaxVesselnessEigenVectors->SetRegions( inputRegion );
    m_MaxVesselnessEigenVectors->Allocate();
  }

  ImageRegionConstIterator< TensorImageType > tensorIterator( inputTensor, inputRegion );
  ImageRegionIteratorWithIndex< InternalImageType > responseIterator( m_MaxVesselnessResponse, inputRegion );
  ImageRegionIterator< VectorImageType > eigenvaluesIterator( m_MaxVesselnessEigenValues, inputRegion );
  ImageRegionIterator< MatrixImageType > eigenvectorsIterator( m_MaxVesselnessEigenVectors, inputRegion );

  while ( !tensorIterator.IsAtEnd() )
    {

    vnl_matrix< Precision > tensorPixel( 3, 3 );
    for ( unsigned int d = 0; d < 3; ++d )
      {
        for ( unsigned int d2 = 0; d2 < 3; ++d2 ) {

          tensorPixel( d, d2 ) = tensorIterator.Get()( d, d2 );

        }
      }

    vnl_symmetric_eigensystem< Precision > eigenSystem( tensorPixel );
    vnl_vector< Precision > eigenvalues( 3 );

    eigenvalues[ 0 ] = eigenSystem.get_eigenvalue( 0 );
    eigenvalues[ 1 ] = eigenSystem.get_eigenvalue( 1 );
    eigenvalues[ 2 ] = eigenSystem.get_eigenvalue( 2 );

    if ( abs( eigenvalues[ 0 ] ) > abs( eigenvalues[ 1 ] ) ) std::swap( eigenvalues[ 0 ], eigenvalues[ 1 ] );
    if ( abs( eigenvalues[ 1 ] ) > abs( eigenvalues[ 2 ] ) ) std::swap( eigenvalues[ 1 ], eigenvalues[ 2 ] );
    if ( abs( eigenvalues[ 0 ] ) > abs( eigenvalues[ 1 ] ) ) std::swap( eigenvalues[ 0 ], eigenvalues[ 1 ] );

    Precision vesselness = VesselnessFunction( eigenvalues );

    if ( firstTime || vesselness > responseIterator.Value() )
      {

      for ( unsigned int d = 0; d < 3; ++d )
        {

        eigenvaluesIterator.Value()[ d ] = eigenvalues[ d ];

        for ( unsigned int d2 = 0; d2 < 3; ++d2 )
          {

          eigenvectorsIterator.Value()( d2, d ) = eigenSystem.get_eigenvector( d )( d2 );

          }
        }

      responseIterator.Set( vesselness );

      }

    ++tensorIterator;
    ++responseIterator;
    ++eigenvaluesIterator;
    ++eigenvectorsIterator;

    }

}


template < class TInputImage, class TOutputImage, class TSmootherType >
void
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::GenerateDiffusionTensor()
{

  ImageRegionType imageRegion = m_MaxVesselnessResponse->GetLargestPossibleRegion();

  typename TensorImageType::Pointer diffusionTensor = TensorImageType::New();
  diffusionTensor->SetRegions( imageRegion );
  diffusionTensor->Allocate();

  ImageRegionIterator< TensorImageType > tensorIterator( diffusionTensor, imageRegion );
  ImageRegionConstIterator< InternalImageType > responseIterator( m_MaxVesselnessResponse, imageRegion );
  ImageRegionConstIterator< VectorImageType > eigenvaluesIterator( m_MaxVesselnessEigenValues, imageRegion );
  ImageRegionConstIterator< MatrixImageType > eigenvectorsIterator( m_MaxVesselnessEigenVectors, imageRegion );

  Precision V;

  MatrixType Q, D, Qt, temp, T;


  while( !tensorIterator.IsAtEnd() )
    {

    V = pow( responseIterator.Value(), 1. / m_Sensitivity );

    if ( V > 0 )
      {

      D.Fill(0);
      for ( unsigned int d = 0; d < 3; ++d )
        {

        if ( d == 2 ) D( d, d ) = 1. + ( m_Omega - 1. ) * V;
        else D( d, d ) = 1. + ( m_Epsilon - 1. ) * V;

        for ( unsigned int d2 = 0; d2 < 3; ++d2 ) Q( d, d2 ) = eigenvectorsIterator.Get()( d, d2 );

        }

        Qt = Q.GetTranspose();

        temp = Q * D;
        T = temp * Qt;

        for ( unsigned int d = 0; d < 3; ++d )
          {
          for ( unsigned int d2 = d; d2 < 3; ++d2 )
            {
            tensorIterator.Value()( d, d2 ) = T( d, d2 );
            }
          }

        }
      else
        {

        tensorIterator.Value().Fill( 0. );
        for ( unsigned int d = 0; d < 3; ++d )
          {
          tensorIterator.Value()( d, d ) = 1.;
          }

        }


      ++tensorIterator;
      ++responseIterator;
      ++eigenvaluesIterator;
      ++eigenvectorsIterator;

   }

   m_DiffusionTensor = diffusionTensor;

}


template < class TInputImage, class TOutputImage, class TSmootherType >
typename VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >::InternalImageType::Pointer
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::DiffusionStep( const InternalImageType * inputImage ) const {

  typename MADFilterType::Pointer diffusionFilter = MADFilterType::New();

  diffusionFilter->SetVerbose( m_Verbose );
  diffusionFilter->SetDiffusionTensor( m_DiffusionTensor );
  diffusionFilter->SetInput( inputImage );
  diffusionFilter->SetTimeStep( m_TimeStep );
  diffusionFilter->SetTolerance( m_Tolerance );
  diffusionFilter->SetNumberOfSteps( m_DiffusionIterations );
  diffusionFilter->SetIterationsPerGrid( m_DiffusionIterationsPerGrid );
  diffusionFilter->SetCycle( m_Cycle );
  diffusionFilter->SetMaxCycles( 100 );

  diffusionFilter->Update();

  return diffusionFilter->GetOutput();

}


template < class TInputImage, class TOutputImage, class TSmootherType >
VEDMultigridImageFilter< TInputImage, TOutputImage, TSmootherType >
::~VEDMultigridImageFilter()
{


};


} // end namespace itk

#endif  /* __itkVEDMultigridImageFilter_hxx */
