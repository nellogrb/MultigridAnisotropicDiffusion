#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "itkNeighborhood.h"
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageDuplicator.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkChangeInformationImageFilter.h"

#include "itkVEDMultigridImageFilter.h"
#include "mad/itkMultigridGaussSeidelSmoother.h"


int itkVEDTest_GS( int argc, char * argv[] )
{

  (void)argc; // to suppress warning

  // Test 1
  std::string inputFilename = "test_data/ved_test.mhd";
  std::string outputFilename = "test_data/ved_test_out.mhd";

  /*std::string inputFilename = "test_data/ved_test2.mhd";
  std::string outputFilename = "test_data/ved_test2_out.mhd";*/

  typedef itk::Image< short, 3 > ImageType;

  typedef itk::ImageFileReader< ImageType > ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFilename);
  reader->Update();

  typename ImageType::Pointer input = ImageType::New();
  input->Graft(reader->GetOutput());

  std::cout << * input << std::endl;
  std::cout << input->GetLargestPossibleRegion() << std::endl;

  typedef itk::mad::MultigridGaussSeidelSmoother< ImageType::ImageDimension > smootherType;
  typedef itk::VEDMultigridImageFilter< ImageType, ImageType, smootherType > filterType;


  filterType::Pointer filter = filterType::New();

  typedef filterType::CycleType cycleType;
  cycleType cycle = filterType::CycleType::VCYCLE;

  if ( std::strcmp( argv[ 1 ], "fmg" ) == 0 )
    cycle = filterType::CycleType::FMG;
  else if ( std::strcmp( argv[ 1 ], "s" ) == 0 )
    cycle = filterType::CycleType::SMOOTHER;

  filter->SetCycle( cycle );
  filter->SetDiffusionIterationsPerGrid( 3 );

  filter->SetInput( input );
  filter->SetVerbose( true );



  std::vector< double > sigmaValues( 5 );

  sigmaValues[0] = 0.300;
  sigmaValues[1] = 0.482;
  sigmaValues[2] = 0.775;
  sigmaValues[3] = 1.245;
  sigmaValues[4] = 2.000;


  filter->SetScales( sigmaValues );
  filter->SetAlpha( 0.5 );
  filter->SetBeta( 0.5 );
  filter->SetGamma( 5. );
  filter->SetEpsilon( 0.01 );
  filter->SetSensitivity( 10. );

  filter->SetIterations( 1 );
  filter->SetTolerance( 1e-10 );

  filter->SetTimeStep( 0.1 );
  filter->SetDiffusionIterations( 4 );


  // Test 1
  filter->SetOmega( 1.5 );

  // Test 2
  //filter->SetOmega( 1.2 );



  filter->Update();


  std::cout << * filter->GetOutput() << std::endl;
  std::cout << filter->GetOutput()->GetLargestPossibleRegion() << std::endl;


  typedef itk::ImageDuplicator< ImageType > DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(filter->GetOutput());
  duplicator->Update();
  typename ImageType::Pointer alignedOutput = duplicator->GetOutput();

  typedef itk::ChangeInformationImageFilter< ImageType > ChangeInformationType;
  typename ChangeInformationType::Pointer changeInformation = ChangeInformationType::New();
  changeInformation->SetInput( alignedOutput );
  changeInformation->SetOutputDirection( input->GetDirection() );
  changeInformation->ChangeDirectionOn();
  changeInformation->UpdateOutputInformation();


  typedef itk::ImageFileWriter< ImageType > WriterType;

  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputFilename);
  writer->SetInput(changeInformation->GetOutput());
  writer->Update();

  return EXIT_SUCCESS;
}
