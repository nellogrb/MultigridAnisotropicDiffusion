set(DOCUMENTATION "This module contains an implementation of a multigrid method to solve a generic anisotropic diffusion problem.")

itk_module(ITKMultigridAnisotropicDiffusion
  DEPENDS
    ITKCommon
    ITKIOImageBase
    ITKImageFilterBase
    ITKImageGrid
    ITKImageFeature
  TEST_DEPENDS
    ITKTestKernel
  EXCLUDE_FROM_DEFAULT
  DESCRIPTION
    "${DOCUMENTATION}"
)
