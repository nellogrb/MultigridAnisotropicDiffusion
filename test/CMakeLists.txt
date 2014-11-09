itk_module_test()

#${itk-module} will be the name of this module and will not need to be
#changed when this module is renamed.

set(${itk-module}Tests
  itk2DDiffusionTest_GS.cxx
  itk2DDiffusionTest_WJ.cxx
  itkVEDTest_GS.cxx
)

CreateTestDriver(${itk-module}  "${${itk-module}-Test_LIBRARIES}" "${${itk-module}Tests}")

itk_add_test(NAME itk2DDiffusionTest_GS_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS fmg)

itk_add_test(NAME itk2DDiffusionTest_GS_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS v)


itk_add_test(NAME itk2DDiffusionTest_WJ_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ fmg)

itk_add_test(NAME itk2DDiffusionTest_WJ_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ v)


itk_add_test(NAME itk2DDiffusionTest_GS_S
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS s)

itk_add_test(NAME itk2DDiffusionTest_WJ_S
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ s)




itk_add_test(NAME itkVEDTest_GS_V
  COMMAND ${itk-module}TestDriver itkVEDTest_GS v)

itk_add_test(NAME itkVEDTest_GS_FMG
  COMMAND ${itk-module}TestDriver itkVEDTest_GS fmg)

itk_add_test(NAME itkVEDTest_GS_S
  COMMAND ${itk-module}TestDriver itkVEDTest_GS s)
