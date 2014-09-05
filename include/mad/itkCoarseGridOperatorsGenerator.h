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
#ifndef __itkCoarseGridOperatorsGenerator_h
#define __itkCoarseGridOperatorsGenerator_h

#include "itkSymmetricSecondRankTensor.h"

#include "itkGridsHierarchy.h"
#include "itkInterGridOperators.h"
#include "itkStencilImage.h"

namespace itk
{

namespace mad
{

/** \class CoarseGridOperatorsGenerator
 *
 * \brief Implementation of the two main generating rules for coarse
 * operators: DCA for Geometric MG, and GCA for algebraic MG.
 *
 * In the first case the coarse operator \f$ A_h, A_{2h}, A_{4h}, \dots \f$
 * are obtained, after the diffusion tensor coefficients have been interpolated
 * on the coarse grids, using the following centered finite differences scheme
 *
 * \f{eqnarray*}
 *    & I^{n+1}(\mathbf{x}) - \Delta t \sum \limits_{r=1}^d
 *    \left( \sum \limits_{s=1}^d \frac{m_{sr}(\mathbf{x} + \mathbf{e}_s)
 *    - m_{sr}(\mathbf{x} - \mathbf{e}_s)}{2h_s} \right) \frac{I^{n+1}(\mathbf{x}
 *    + \mathbf{e}_r) - I^{n+1}(\mathbf{x} - \mathbf{e}_r)}{2h_r} \\
 *   & - \Delta t \sum \limits_{s \neq r,\, s,r=1}^d m_{sr}(\mathbf{x}) \frac{I^{n+1}(\mathbf{x}
 *    + \mathbf{e}_s + \mathbf{e}_r) - I^{n+1}(\mathbf{x} + \mathbf{e}_s - \mathbf{e}_r) -
 *    I^{n+1}(\mathbf{x} - \mathbf{e}_s + \mathbf{e}_r) + I^{n+1}(\mathbf{x} -
 *    \mathbf{e}_s - \mathbf{e}_r)  }{4h_s h_r} \\
 *   & - \Delta t \sum \limits_{r=1}^d m_{rr}(\mathbf{x}) \frac{I^{n+1}(\mathbf{x} + \mathbf{e}_r )
 *   - 2I^{n+1}(\mathbf{x}) + I^{n+1}(\mathbf{x} - \mathbf{e}_r ) }{h^2_r} = I^n(\mathbf{x}),
 * \f}
 *
 * where \f$ \mathbf{x} \f$ is a generic interior point, and \f$ \mathbf{e}_p \f$
 * is the versor in the \f$ p \f$ direction; hence, in the different grids, the same discretization
 * is applied with increasing spatial steps. The original problem is described in class
 * MultigridAnisotropicDiffusionImageFilter. Regarding the border points, a homogeneous
 * Neumann condition is considered, and implemented by forcing the virtual points outside
 * the grid to assume the same value as their correspondent symmetric with respect to
 * the border. On the other hand, for the tensor's coefficients derivatives, the central
 * difference is substituted by a second order forward/backward finite difference.
 *
 * In the second case, \f$ A_{2h} \f$ is defined as follows
 *
 * \f[
 *  A_{2h} = R_h^{2h} A_h I_{2h}^h
 * \f]
 *
 * where \f$ A_h \f$ is the operator on the finer grid. As in this MG
 * module we do not explicitly build matrix and vectors, this is done
 * using a superposition principle: the coefficients of \f$ A_{2h} \f$
 * are obtained with the summation of effects \f$ A_{2h} = \sum \limits_i A^i_{2h} \f$,
 * where
 *
 * \f[
 *  A^i_{2h} = R_h^{2h} A_h I_{2h}^h \mathbf{y}_i
 * \f]
 *
 * and \f$ \mathbf{y}_i \f$ is an image filled with zeros, except for an entry
 * 1 corresponding to a single pixel; this is hence iterated for every
 * pixel in the region to finally get \f$ A_{2h} \f$.
 * This definition is recursive, starting with the operator at level 0
 * which is built with the above finite differences scheme.
 *
 * \author Antonello Gerbi
 *
 * \ingroup ITKMultigridAnisotropicDiffusion
 */

template < unsigned int VDimension >
class ITK_EXPORT CoarseGridOperatorsGenerator
{
public:

  /** Standard class typedefs. */
  typedef double                                                           Precision;
  typedef CoarseGridOperatorsGenerator                                    Self;

  typedef Image< Precision, VDimension >                                  ImageType;
  typedef Neighborhood< Precision, VDimension >                           StencilType;
  typedef StencilImage< Precision, VDimension >                           StencilImageType;

  typedef SymmetricSecondRankTensor< Precision, VDimension >              TensorType;
  typedef Image< TensorType, VDimension >                                 TensorImageType;

  typedef ImageRegion< VDimension >                                       ImageRegionType;
  typedef typename ImageType::SpacingType                                 SpacingType;
  typedef typename ImageType::SizeType                                    SizeType;
  typedef typename ImageType::IndexType                                   IndexType;
  typedef typename ImageType::OffsetType                                  OffsetType;
  typedef typename std::list< OffsetType >                                OffsetListType;

  typedef mad::InterGridOperators< VDimension >                           InterGridOperatorsType;
  typedef typename InterGridOperatorsType::PointPositionType              PointPositionType;

  enum CoarseGridOperatorType { DCA, GCA };

  /** Class constructor which takes as argument the coarse grid
   *  operator generation rule. */
  CoarseGridOperatorsGenerator( const CoarseGridOperatorType & operatorType );

  /** Class destructor. */
  ~CoarseGridOperatorsGenerator() {};

  /** Main class member, which fills the pointed class gridsHierarchy, at
   *  each level, using the prescribed operator generation rule. It also needs
   *  the tensor on the finest region and the timeStep, in order to calculate
   *  the operator coefficients.  */
  void GenerateOperators( GridsHierarchy< VDimension > * gridsHierarchy, const TensorImageType * fineTensor,
                          const Precision timeStep ) const;

private:

  CoarseGridOperatorType                                 m_CoarseGridOperator;

  /** Direct Coarse Approximation implementation. */
  void GenerateDCA( typename GridsHierarchy< VDimension >::Grid * grid,
                    const TensorImageType * tensor,
                    const Precision timeStep ) const;

  /** Galerkin Coarse Approximation implementation. */
  void GenerateGCA( typename GridsHierarchy< VDimension >::Grid * fineGrid,
                    typename GridsHierarchy< VDimension >::Grid * coarseGrid ) const;


  /** Purposely not implemented */
  CoarseGridOperatorsGenerator( const Self & );
  void operator=( const Self & );

};


} // end namespace mad

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCoarseGridOperatorsGenerator.hxx"
#endif

#endif /* __itkCoarseGridOperatorsGenerator_h */
