/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/algorithms/Aggregates.h

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

#include <functional>
#include <utility>

#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>

NAMESPACE_BEGIN(Grid);

inline RealD AggregatePowerLaw(RealD x)
{
  //  return std::pow(x,-4);
  //  return std::pow(x,-3);
  return std::pow(x,-5);
}

// Project an operator application into a constrained fine-field subspace.
// The fixed temporal Wilson-clover boundary requires setup CG to remain in
// the endpoint-projected space, not merely to receive a projected result after
// a complete solve.  Wrapping HermOp here projects both inputs and outputs of
// every iteration performed by the generic Krylov solver.
template<class Field>
class ProjectedLinearOperator : public LinearOperatorBase<Field> {
public:
  using Projector = std::function<void(Field&)>;

  ProjectedLinearOperator(LinearOperatorBase<Field> &wrapped,
                          Projector projector)
    : wrapped_(wrapped), projector_(std::move(projector))
  {}

  void projectInput(const Field &in, Field &projected) const {
    projected = in;
    if (projector_) projector_(projected);
  }

  void projectOutput(Field &out) const {
    if (projector_) projector_(out);
  }

  void OpDiag(const Field &in, Field &out) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.OpDiag(projected, out);
    projectOutput(out);
  }

  void OpDir(const Field &in, Field &out, int dir, int disp) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.OpDir(projected, out, dir, disp);
    projectOutput(out);
  }

  void OpDirAll(const Field &in, std::vector<Field> &out) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.OpDirAll(projected, out);
    for (auto &field: out) projectOutput(field);
  }

  void Op(const Field &in, Field &out) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.Op(projected, out);
    projectOutput(out);
  }

  void AdjOp(const Field &in, Field &out) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.AdjOp(projected, out);
    projectOutput(out);
  }

  void HermOpAndNorm(const Field &in, Field &out,
                     RealD &n1, RealD &n2) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.HermOp(projected, out);
    projectOutput(out);
    n1 = real(innerProduct(projected, out));
    n2 = norm2(out);
  }

  void HermOp(const Field &in, Field &out) override {
    Field projected(in.Grid());
    projectInput(in, projected);
    wrapped_.HermOp(projected, out);
    projectOutput(out);
  }

private:
  LinearOperatorBase<Field> &wrapped_;
  Projector projector_;
};

template<class Fobj,class CComplex,int nbasis>
class Aggregation {
public:
  constexpr int Nbasis(void) { return nbasis; };
  
  typedef iVector<CComplex,nbasis >             siteVector;
  typedef Lattice<siteVector>                 CoarseVector;
  typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;

  typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
  typedef Lattice<Fobj >        FineField;
  using Projector = std::function<void(FineField&)>;

  GridBase *CoarseGrid;
  GridBase *FineGrid;
  std::vector<Lattice<Fobj> > subspace;
  int checkerboard;
  Projector projector;
  int Checkerboard(void){return checkerboard;}
  Aggregation(GridBase *_CoarseGrid,GridBase *_FineGrid,int _checkerboard,
              Projector _projector = Projector()) :
    CoarseGrid(_CoarseGrid),
    FineGrid(_FineGrid),
    subspace(nbasis,_FineGrid),
    checkerboard(_checkerboard),
    projector(std::move(_projector))
  {
  };
  
  void Project(FineField &field) const {
    if (projector) projector(field);
  }

  void ValidateFinite(const FineField &field, const char *stage) const {
    const RealD n2 = norm2(field);
    if (!std::isfinite(n2)) {
      std::cout << GridLogError << "Non-finite aggregation field at "
                << stage << std::endl;
      GRID_ASSERT(0);
    }
  }

  void ValidateProjected(const FineField &field, const char *stage,
                         RealD tolerance = 1.0e-13) const {
    ValidateFinite(field, stage);
    if (!projector) return;
    FineField projected(FineGrid);
    FineField difference(FineGrid);
    projected = field;
    Project(projected);
    difference = field - projected;
    const RealD fieldNorm2 = norm2(field);
    const RealD differenceNorm2 = norm2(difference);
    const RealD relative = (fieldNorm2 == 0.0)
      ? std::sqrt(differenceNorm2)
      : std::sqrt(differenceNorm2 / fieldNorm2);
    if (!std::isfinite(relative) || relative > tolerance) {
      std::cout << GridLogError << "Aggregation field left projected space at "
                << stage << ": relative endpoint norm " << relative
                << std::endl;
      GRID_ASSERT(0);
    }
  }

  void GlobalOrthonormalise(void)
  {
    // Normalise all vectors
    for(int i=0;i<nbasis; i++){
      Project(subspace[i]);
      const RealD n2 = norm2(subspace[i]);
      GRID_ASSERT(std::isfinite(n2) && n2 > 0.0);
      RealD scale = std::pow(n2,-0.5);
      subspace[i] = subspace[i]*scale;
    }
    for(int i=0;i<nbasis; i++){
      for(int j=0;j<i; j++){
	basisOrthogonalize(subspace,subspace[i],j);
      }
      Project(subspace[i]);
      const RealD n2 = norm2(subspace[i]);
      GRID_ASSERT(std::isfinite(n2) && n2 > 0.0);
      RealD scale = std::pow(n2,-0.5);
      subspace[i] = subspace[i]*scale;
    }
  }  
  void Orthogonalise(void){
    CoarseScalar InnerProd(CoarseGrid); 
    //    std::cout << GridLogMessage <<" Block Gramm-Schmidt pass 1"<<std::endl;
    blockOrthogonalise(InnerProd,subspace);
    for (auto &basis: subspace) {
      Project(basis);
      ValidateProjected(basis, "block orthonormalization");
    }
  } 
  void ProjectToSubspace(CoarseVector &CoarseVec,const FineField &FineVec){
    FineField projected(FineGrid);
    projected = FineVec;
    Project(projected);
    blockProject(CoarseVec,projected,subspace);
  }
  void PromoteFromSubspace(const CoarseVector &CoarseVec,FineField &FineVec){
    FineVec.Checkerboard() = subspace[0].Checkerboard();
    blockPromote(CoarseVec,FineVec,subspace);
    Project(FineVec);
  }

  virtual void CreateSubspaceRandom(GridParallelRNG  &RNG) {
    int nn=nbasis;
    RealD scale;
    FineField noise(FineGrid);
    for(int b=0;b<nn;b++){
      subspace[b] = Zero();
      gaussian(RNG,noise);
      Project(noise);
      const RealD noiseNorm2 = norm2(noise);
      GRID_ASSERT(std::isfinite(noiseNorm2) && noiseNorm2 > 0.0);
      scale = std::pow(noiseNorm2,-0.5);
      noise=noise*scale;
      subspace[b] = noise;
      ValidateProjected(subspace[b], "random subspace generation");
    }
  }
  virtual void CreateSubspace(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis)
  {

    RealD scale;

    ConjugateGradient<FineField> CG(1.0e-4,2000,true);
    ProjectedLinearOperator<FineField> projectedHermop(hermop, projector);
    FineField noise(FineGrid);
    FineField Mn(FineGrid);

    for(int b=0;b<nn;b++){
      
      subspace[b] = Zero();
      gaussian(RNG,noise);
      Project(noise);
      RealD noiseNorm2 = norm2(noise);
      GRID_ASSERT(std::isfinite(noiseNorm2) && noiseNorm2 > 0.0);
      scale = std::pow(noiseNorm2,-0.5);
      noise=noise*scale;
      
      projectedHermop.HermOp(noise,Mn); std::cout<<GridLogMessage << "noise   ["<<b<<"] <n|MdagM|n> "<<norm2(Mn)<<std::endl;

      for(int i=0;i<2;i++){

	CG(projectedHermop,noise,subspace[b]);
	Project(subspace[b]);

	noise = subspace[b];
	noiseNorm2 = norm2(noise);
	GRID_ASSERT(std::isfinite(noiseNorm2) && noiseNorm2 > 0.0);
	scale = std::pow(noiseNorm2,-0.5);
	noise=noise*scale;
	
	projectedHermop.HermOp(noise,Mn); std::cout<<GridLogMessage << "intermediate["<<i<<"] <i|MdagM|i> "<<norm2(Mn)<<std::endl;
	
      }

      projectedHermop.HermOp(noise,Mn); std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(Mn)<<std::endl;
      subspace[b]   = noise;
      ValidateProjected(subspace[b], "near-null setup");

    }
  }

  virtual void CreateSubspaceGCR(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &DiracOp,int nn=nbasis)
  {
    RealD scale;

    TrivialPrecon<FineField> simple_fine;
    //    PrecGeneralisedConjugateResidualNonHermitian<FineField> GCR(0.001,10,DiracOp,simple_fine,30,30);
    //    PrecGeneralisedConjugateResidualNonHermitian<FineField> GCR(0.001,10,DiracOp,simple_fine,12,12);
    //    PrecGeneralisedConjugateResidualNonHermitian<FineField> GCR(0.001,30,DiracOp,simple_fine,12,12);
    //    PrecGeneralisedConjugateResidualNonHermitian<FineField> GCR(0.0005,30,DiracOp,simple_fine,20,20);
    PrecGeneralisedConjugateResidualNonHermitian<FineField> GCR(0.0005,30,DiracOp,simple_fine,10,10);
    FineField noise(FineGrid);
    FineField src(FineGrid);
    FineField guess(FineGrid);
    FineField Mn(FineGrid);

    for(int b=0;b<nn;b++){
      
      subspace[b] = Zero();
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;
      
      DiracOp.Op(noise,Mn); std::cout<<GridLogMessage << "noise   ["<<b<<"] <n|Op|n> "<<innerProduct(noise,Mn)<<std::endl;

      for(int i=0;i<3;i++){
	//  void operator() (const Field &src, Field &psi){
#if 1
	if (i==0)std::cout << GridLogMessage << " inverting on noise "<<std::endl;
	src = noise;
	guess=Zero();
	GCR(src,guess);
	subspace[b] = guess;
#else
	if (i==0)std::cout << GridLogMessage << " inverting on zero "<<std::endl;
	src=Zero();
	guess = noise;
	GCR(src,guess);
	subspace[b] = guess;
#endif
	noise = subspace[b];
	scale = std::pow(norm2(noise),-0.5); 
	noise=noise*scale;
	
	DiracOp.Op(noise,Mn); std::cout<<GridLogMessage << "intermediate["<<i<<"] <f|Op|f> "<<innerProduct(noise,Mn)<<" <f|OpDagOp|f>"<<norm2(Mn)<<std::endl;

      }

      DiracOp.Op(noise,Mn); std::cout<<GridLogMessage << "filtered["<<b<<"] <f|Op|f> "<<innerProduct(noise,Mn)<<" <f|OpDagOp|f>"<<norm2(Mn)<<std::endl;
      subspace[b]   = noise;

    }
    GlobalOrthonormalise();
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // World of possibilities here. But have tried quite a lot of experiments (250+ jobs run on Summit)
  // and this is the best I found
  ////////////////////////////////////////////////////////////////////////////////////////////////

  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo,
				       int orderfilter,
				       int ordermin,
				       int orderstep,
				       double filterlo
				       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    gaussian(RNG,noise);
    scale = std::pow(norm2(noise),-0.5); 
    noise=noise*scale;

    std::cout << GridLogMessage<<" Chebyshev subspace pass-1 : ord "<<orderfilter<<" ["<<lo<<","<<hi<<"]"<<std::endl;
    std::cout << GridLogMessage<<" Chebyshev subspace pass-2 : nbasis"<<nn<<" min "
	      <<ordermin<<" step "<<orderstep
	      <<" lo"<<filterlo<<std::endl;

    // Initial matrix element
    hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

    int b =0;
    {
      ComplexD ip;
      // Filter
      Chebyshev<FineField> Cheb(lo,hi,orderfilter);
      Cheb(hermop,noise,Mn);
      // normalise
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;
      subspace[b]   = Mn;

      hermop.Op(Mn,tmp);
      ip= innerProduct(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|Op|n> "<<norm2(tmp)<<" "<<ip<<std::endl;

      hermop.AdjOp(Mn,tmp); 
      ip = innerProduct(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|AdjOp|n> "<<norm2(tmp)<<" "<<ip<<std::endl;
      b++;
    }

    // Generate a full sequence of Chebyshevs
    {
      lo=filterlo;
      noise=Mn;

      FineField T0(FineGrid); T0 = noise;  
      FineField T1(FineGrid); 
      FineField T2(FineGrid);
      FineField y(FineGrid);
      
      FineField *Tnm = &T0;
      FineField *Tn  = &T1;
      FineField *Tnp = &T2;

      // Tn=T1 = (xscale M + mscale)in
      RealD xscale = 2.0/(hi-lo);
      RealD mscale = -(hi+lo)/(hi-lo);
      hermop.HermOp(T0,y);
      T1=y*xscale+noise*mscale;

      for(int n=2;n<=ordermin+orderstep*(nn-2);n++){
	
	hermop.HermOp(*Tn,y);

	autoView( y_v , y, AcceleratorWrite);
	autoView( Tn_v , (*Tn), AcceleratorWrite);
	autoView( Tnp_v , (*Tnp), AcceleratorWrite);
	autoView( Tnm_v , (*Tnm), AcceleratorWrite);
	const int Nsimd = CComplex::Nsimd();
	accelerator_for(ss, FineGrid->oSites(), Nsimd, {
	  coalescedWrite(y_v[ss],xscale*y_v(ss)+mscale*Tn_v(ss));
	  coalescedWrite(Tnp_v[ss],2.0*y_v(ss)-Tnm_v(ss));
        });

	// Possible more fine grained control is needed than a linear sweep,
	// but huge productivity gain if this is simple algorithm and not a tunable
	int m =1;
	if ( n>=ordermin ) m=n-ordermin;
	if ( (m%orderstep)==0 ) { 
	  Mn=*Tnp;
	  scale = std::pow(norm2(Mn),-0.5);         Mn=Mn*scale;
	  subspace[b] = Mn;


	  ComplexD ip;

	  hermop.Op(Mn,tmp);
	  ip= innerProduct(Mn,tmp); 
	  std::cout<<GridLogMessage << "filt ["<<b<<"] <n|Op|n> "<<norm2(tmp)<<" "<<ip<<std::endl;

	  hermop.AdjOp(Mn,tmp); 
	  ip = innerProduct(Mn,tmp); 
	  std::cout<<GridLogMessage << "filt ["<<b<<"] <n|AdjOp|n> "<<norm2(tmp)<<" "<<ip<<std::endl;
	  
	  b++;
	}

	// Cycle pointers to avoid copies
	FineField *swizzle = Tnm;
	Tnm    =Tn;
	Tn     =Tnp;
	Tnp    =swizzle;
	  
      }
    }
    GRID_ASSERT(b==nn);
  }


  virtual void CreateSubspacePolyCheby(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo1,
				       int orderfilter,
				       double lo2,
				       int orderstep)
  {
    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    gaussian(RNG,noise);
    scale = std::pow(norm2(noise),-0.5); 
    noise=noise*scale;

    std::cout << GridLogMessage<<" CreateSubspacePolyCheby "<<std::endl;
    // Initial matrix element
    hermop.Op(noise,Mn);
    std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

    int b =0;
    {
      // Filter
      std::cout << GridLogMessage << "Cheby "<<lo1<<","<<hi<<" "<<orderstep<<std::endl;
      Chebyshev<FineField> Cheb(lo1,hi,orderfilter);
      Cheb(hermop,noise,Mn);
      // normalise
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|n> "<<norm2(Mn)<<std::endl;
    }

    // Generate a full sequence of Chebyshevs
    for(int n=1;n<nn;n++){
      std::cout << GridLogMessage << "Cheby "<<lo2<<","<<hi<<" "<<orderstep<<std::endl;
      Chebyshev<FineField> Cheb(lo2,hi,orderstep);
      Cheb(hermop,subspace[n-1],Mn);

      for(int m=0;m<n;m++){
	ComplexD c = innerProduct(subspace[m],Mn);
	Mn = Mn - c*subspace[m];
      }
      
      // normalise
      scale = std::pow(norm2(Mn),-0.5);
      Mn=Mn*scale;
      
      subspace[n]=Mn;
      
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<n<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      std::cout<<GridLogMessage << "filt ["<<n<<"] <n|n> "<<norm2(Mn)<<std::endl;

    }
  }

  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo,
				       int orderfilter
				       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    std::cout << GridLogMessage<<" Chebyshev subspace pure noise : ord "<<orderfilter<<" ["<<lo<<","<<hi<<"]"<<std::endl;
    std::cout << GridLogMessage<<" Chebyshev subspace pure noise  : nbasis "<<nn<<std::endl;


    for(int b =0;b<nbasis;b++)
    {
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;

      // Initial matrix element
      hermop.Op(noise,Mn);
      if(b==0) std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

      // Filter
      Chebyshev<FineField> Cheb(lo,hi,orderfilter);
      Cheb(hermop,noise,Mn);
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;

      // Refine
      Chebyshev<FineField> PowerLaw(lo,hi,1000,AggregatePowerLaw);
      noise = Mn;
      PowerLaw(hermop,noise,Mn);
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;

      // normalise
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
    }

  }

  virtual void CreateSubspaceChebyshevPowerLaw(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
					       int nn,
					       double hi,
					       int orderfilter
					       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    std::cout << GridLogMessage<<" Chebyshev subspace pure noise : ord "<<orderfilter<<" [0,"<<hi<<"]"<<std::endl;
    std::cout << GridLogMessage<<" Chebyshev subspace pure noise  : nbasis "<<nn<<std::endl;

    for(int b =0;b<nbasis;b++)
    {
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;

      // Initial matrix element
      hermop.Op(noise,Mn);
      if(b==0) std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;
      // Filter
      Chebyshev<FineField> Cheb(0.0,hi,orderfilter,AggregatePowerLaw);
      Cheb(hermop,noise,Mn);
      // normalise
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
    }

  }
  virtual void CreateSubspaceChebyshevNew(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
					  double hi
					  ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    for(int b =0;b<nbasis;b++)
    {
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;

      // Initial matrix element
      hermop.Op(noise,Mn);
      if(b==0) std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;
      // Filter
      //#opt2(x) =  acheb(x,3,90,300)* acheb(x,1,90,50) * acheb(x,0.5,90,200) * acheb(x,0.05,90,400) * acheb(x,0.01,90,1500)
      /*266
      Chebyshev<FineField> Cheb1(3.0,hi,300);
      Chebyshev<FineField> Cheb2(1.0,hi,50);
      Chebyshev<FineField> Cheb3(0.5,hi,300);
      Chebyshev<FineField> Cheb4(0.05,hi,500);
      Chebyshev<FineField> Cheb5(0.01,hi,2000);
      */
      /* 242 */
      /*
      Chebyshev<FineField> Cheb3(0.1,hi,300);
      Chebyshev<FineField> Cheb2(0.02,hi,1000);
      Chebyshev<FineField> Cheb1(0.003,hi,2000);
      8?
      */
      /* How many??
      */
      Chebyshev<FineField> Cheb2(0.001,hi,2500); // 169 iters on HDCG after refine
      Chebyshev<FineField> Cheb1(0.02,hi,600);

      //      Chebyshev<FineField> Cheb2(0.001,hi,1500);
      //      Chebyshev<FineField> Cheb1(0.02,hi,600);
      Cheb1(hermop,noise,Mn); scale = std::pow(norm2(Mn),-0.5); 	noise=Mn*scale;
      hermop.Op(noise,tmp); std::cout<<GridLogMessage << "Cheb1 <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      Cheb2(hermop,noise,Mn); scale = std::pow(norm2(Mn),-0.5); 	noise=Mn*scale;
      hermop.Op(noise,tmp); std::cout<<GridLogMessage << "Cheb2 <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      //      Cheb3(hermop,noise,Mn); scale = std::pow(norm2(Mn),-0.5); 	noise=Mn*scale;
      //      hermop.Op(noise,tmp); std::cout<<GridLogMessage << "Cheb3 <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      //      Cheb4(hermop,noise,Mn); scale = std::pow(norm2(Mn),-0.5); 	noise=Mn*scale;
      //      hermop.Op(noise,tmp); std::cout<<GridLogMessage << "Cheb4 <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      //      Cheb5(hermop,noise,Mn); scale = std::pow(norm2(Mn),-0.5); 	noise=Mn*scale;
      //      hermop.Op(noise,tmp); std::cout<<GridLogMessage << "Cheb5 <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      subspace[b]   = noise;
      hermop.Op(subspace[b],tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<< " norm " << norm2(noise)<<std::endl;
    }

  }

  virtual void CreateSubspaceMultishift(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
					double Lo,double tol,int maxit)
  {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    std::cout << GridLogMessage<<" Multishift subspace : Lo "<<Lo<<std::endl;

    // Filter
    // [ 1/6(x+Lo)  - 1/2(x+2Lo) + 1/2(x+3Lo)  -1/6(x+4Lo) = Lo^3 /[ (x+1Lo)(x+2Lo)(x+3Lo)(x+4Lo) ]
    //
    // 1/(x+Lo)  - 1/(x+2 Lo)
    double epsilon      = Lo/3;
    std::vector<RealD> alpha({1.0/6.0,-1.0/2.0,1.0/2.0,-1.0/6.0});
    std::vector<RealD> shifts({Lo,Lo+epsilon,Lo+2*epsilon,Lo+3*epsilon});
    std::vector<RealD> tols({tol,tol,tol,tol});
    std::cout << "sizes "<<alpha.size()<<" "<<shifts.size()<<" "<<tols.size()<<std::endl;

    MultiShiftFunction msf(4,0.0,95.0);
    std::cout << "msf constructed "<<std::endl;
    msf.poles=shifts;
    msf.residues=alpha;
    msf.tolerances=tols;
    msf.norm=0.0;
    msf.order=alpha.size();
    ConjugateGradientMultiShift<FineField> MSCG(maxit,msf);
    
    for(int b =0;b<nbasis;b++)
    {
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;

      // Initial matrix element
      hermop.Op(noise,Mn);
      if(b==0) std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

      MSCG(hermop,noise,Mn);
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;

    }

  }
  virtual void RefineSubspace(LinearOperatorBase<FineField> &hermop,
			      double Lo,double tol,int maxit)
  {
    FineField tmp(FineGrid);
    for(int b =0;b<nbasis;b++)
    {
      ConjugateGradient<FineField>  CGsloppy(tol,maxit,false);
      ShiftedHermOpLinearOperator<FineField> ShiftedFineHermOp(hermop,Lo);
      tmp=Zero();
      CGsloppy(hermop,subspace[b],tmp);
      RealD scale = std::pow(norm2(tmp),-0.5); 	tmp=tmp*scale;
      subspace[b]=tmp;
      hermop.Op(subspace[b],tmp);
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
    }
  }
  virtual void RefineSubspaceHDCG(LinearOperatorBase<FineField> &hermop,
				  TwoLevelCGmrhs<FineField> & theHDCG,
				  int nrhs)
  {
    std::vector<FineField> src_mrhs(nrhs,FineGrid);
    std::vector<FineField> res_mrhs(nrhs,FineGrid);
    FineField tmp(FineGrid);
    for(int b =0;b<nbasis;b+=nrhs)
    {
      tmp = subspace[b];
      RealD scale = std::pow(norm2(tmp),-0.5); 	tmp=tmp*scale;
      subspace[b] =tmp;
      hermop.Op(subspace[b],tmp);
      std::cout<<GridLogMessage << "before filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;

      for(int r=0;r<MIN(nbasis-b,nrhs);r++){
	src_mrhs[r] = subspace[b+r];
      }
      for(int r=0;r<nrhs;r++){
	res_mrhs[r] = Zero();
      }
      theHDCG(src_mrhs,res_mrhs);

      for(int r=0;r<MIN(nbasis-b,nrhs);r++){
	tmp = res_mrhs[r];
	RealD scale = std::pow(norm2(tmp),-0.5); tmp=tmp*scale;
	subspace[b+r]=tmp;
      }
      hermop.Op(subspace[b],tmp);
      std::cout<<GridLogMessage << "after filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
    }
  }

  
  
};
NAMESPACE_END(Grid);

