itk_module_test()

#${itk-module} will be the name of this module and will not need to be
#changed when this module is renamed.

set(${itk-module}Tests
  itk2DDiffusionTest_GS.cxx
  itk2DDiffusionTest_WJ.cxx
  itkVEDTest_GS.cxx
)

CreateTestDriver(${itk-module}  "${${itk-module}-Test_LIBRARIES}" "${${itk-module}Tests}")

itk_add_test(NAME itk2DDiffusionTest_GS_DCA_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS dca fmg)

itk_add_test(NAME itk2DDiffusionTest_GS_DCA_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS dca v)

itk_add_test(NAME itk2DDiffusionTest_GS_GCA_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS gca fmg)

itk_add_test(NAME itk2DDiffusionTest_GS_GCA_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS gca v)


itk_add_test(NAME itk2DDiffusionTest_WJ_DCA_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ dca fmg)

itk_add_test(NAME itk2DDiffusionTest_WJ_DCA_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ dca v)

itk_add_test(NAME itk2DDiffusionTest_WJ_GCA_FMG
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ gca fmg)

itk_add_test(NAME itk2DDiffusionTest_WJ_GCA_V
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ gca v)


itk_add_test(NAME itk2DDiffusionTest_GS_S
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_GS dca s)

itk_add_test(NAME itk2DDiffusionTest_WJ_S
  COMMAND ${itk-module}TestDriver itk2DDiffusionTest_WJ dca s)




itk_add_test(NAME itkVEDTest_GS_DCA_V
  COMMAND ${itk-module}TestDriver itkVEDTest_GS dca v)

itk_add_test(NAME itkVEDTest_GS_DCA_FMG
  COMMAND ${itk-module}TestDriver itkVEDTest_GS dca fmg)

itk_add_test(NAME itkVEDTest_GS_S
  COMMAND ${itk-module}TestDriver itkVEDTest_GS dca s)
