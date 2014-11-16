#include <iostream>
#include <string>
#include <cmath>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkImageIOBase.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkMultigridAnisotropicDiffusionImageFilter.h"
#include "mad/itkMultigridGaussSeidelSmoother.h"


int itk2DDiffusionTest_GS( int argc, char * argv[] )
{

  (void)argc; // to suppress warning

  std::string inputFilename = "test_data/lena.jpg";
  std::string outputFilename = "test_data/lena_out.jpg";
  std::string outputDifferenceFilename = "test_data/lena_diff.jpg";

  typedef itk::Image< unsigned char, 2 > InputImageType;
  typedef itk::Image< float, 2 > ImageType;

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputFilename );
  reader->Update();

  typename InputImageType::Pointer input = InputImageType::New();
  input->Graft( reader->GetOutput() );


  typedef itk::CastImageFilter< InputImageType, ImageType > CastFilterType;
  CastFilterType::Pointer castFilter = CastFilterType::New();
  castFilter->SetInput( input );
  castFilter->Update();

  typename ImageType::Pointer convertedInput = castFilter->GetOutput();

  std::cout << * convertedInput << std::endl;
  std::cout << convertedInput->GetLargestPossibleRegion() << std::endl;

  typedef itk::mad::MultigridGaussSeidelSmoother< ImageType::ImageDimension > smootherType;
  typedef itk::MultigridAnisotropicDiffusionImageFilter< ImageType, ImageType, smootherType > filterType;

  typedef filterType::InputTensorImageType tensorImageType;
  typename tensorImageType::Pointer tensorImage = tensorImageType::New();

  tensorImage->SetRegions( input->GetLargestPossibleRegion() );
  tensorImage->Allocate();

  itk::ImageRegionIteratorWithIndex< tensorImageType > tensorIterator( tensorImage, input->GetLargestPossibleRegion() );

  //typename ImageType::IndexType index;
  //typename ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();

  while ( !tensorIterator.IsAtEnd() )
    {


    // Space-independent anisotropic tensor
    tensorIterator.Value()( 0, 0 ) = 50.;

    tensorIterator.Value()( 1, 1 ) = 30.;

    tensorIterator.Value()( 0, 1 ) = 0.;


    // Space-dependent isotropic tensor

    //index = tensorIterator.GetIndex();

    /*tensorIterator.Value()( 0, 0 ) = 50. + 50. * ( cos( ( 4 * M_PI ) * (float) index[ 0 ] / (float) size[ 0 ] ) ) ;

    tensorIterator.Value()( 1, 1 ) = tensorIterator.Value()( 0, 0 );

    tensorIterator.Value()( 0, 1 ) = 0.;*/


    ++tensorIterator;

    }

  filterType::Pointer filter = filterType::New();
  filter->SetInput( convertedInput );
  filter->SetDiffusionTensor( tensorImage );

  filter->SetIterationsPerGrid( 2 );
  filter->SetVerbose( true );
  filter->SetTimeStep( 0.1 );
  filter->SetNumberOfSteps( 1 );
  filter->SetMaxCycles( 100 );
  filter->SetTolerance( 1e-10 );

  typedef filterType::CycleType cycleType;
  cycleType cycle = filterType::CycleType::VCYCLE;

  if ( std::strcmp( argv[ 1 ], "fmg" ) == 0 )
    cycle = filterType::CycleType::FMG;
  else if ( std::strcmp( argv[ 1 ], "s" ) == 0 )
    cycle = filterType::CycleType::SMOOTHER;

  filter->SetCycle( cycle );

  filter->Update();


  typename ImageType::Pointer differenceImage = ImageType::New();
  differenceImage->SetRegions( input->GetLargestPossibleRegion() );
  differenceImage->Allocate();
  differenceImage->SetSpacing( input->GetSpacing() );


  itk::ImageRegionIteratorWithIndex< ImageType > differenceIterator( differenceImage, differenceImage->GetLargestPossibleRegion() );

  while ( !differenceIterator.IsAtEnd() )
    {

    differenceIterator.Set( abs( convertedInput->GetPixel( differenceIterator.GetIndex() )
                                 - filter->GetOutput()->GetPixel( differenceIterator.GetIndex() ) ) );

    ++differenceIterator;

    }


  typedef itk::CastImageFilter< ImageType, InputImageType > CastBackFilterType;
  CastBackFilterType::Pointer castBackFilter = CastBackFilterType::New();
  castBackFilter->SetInput( filter->GetOutput() );
  castBackFilter->Update();


  typedef itk::ImageFileWriter< InputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( outputFilename );
  writer->SetInput( castBackFilter->GetOutput() );
  writer->Update();


  castBackFilter->SetInput( differenceImage );
  castBackFilter->Update();

  writer->SetFileName( outputDifferenceFilename );
  writer->SetInput( castBackFilter->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}
