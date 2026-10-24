/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/qcd/utils/WilsonLoops.h

    Copyright (C) 2015

    Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
    Author: Peter Boyle <paboyle@ph.ed.ac.uk>
    Author: neo <cossu@post.kek.jp>
    Author: paboyle <paboyle@ph.ed.ac.uk>
    Author: James Harrison <J.Harrison@soton.ac.uk>
    Author: Antonin Portelli <antonin.portelli@me.com>

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

    See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
#ifndef QCD_UTILS_WILSON_LOOPS_H
#define QCD_UTILS_WILSON_LOOPS_H

#include <Grid/qcd/utils/GaugeLoopPath.h>

NAMESPACE_BEGIN(Grid);

// Common wilson loop observables
template <class Gimpl> class WilsonLoops : public Gimpl {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename Gimpl::GaugeField GaugeLorentz;

  typedef GaugeLoopPath<Gimpl> LoopPath;

  //////////////////////////////////////////////////
  // directed plaquette oriented in mu,nu plane
  //////////////////////////////////////////////////
  static void dirPlaquette(GaugeMat &plaq, const std::vector<GaugeMat> &U,
                           const int mu, const int nu) {
    // Annoyingly, must use either scope resolution to find dependent base
    // class,
    // or this-> ; there is no "this" in a static method. This forces explicit
    // Gimpl scope
    // resolution throughout the usage in this file, and rather defeats the
    // purpose of deriving
    // from Gimpl.
    /*
      plaq = Gimpl::CovShiftBackward(
      U[mu], mu, Gimpl::CovShiftBackward(
      U[nu], nu, Gimpl::CovShiftForward(U[mu], mu, U[nu])));
    */
    // _
    //|< _|
    plaq = Gimpl::CovShiftForward(U[mu],mu,
				  Gimpl::CovShiftForward(U[nu],nu,
							 Gimpl::CovShiftBackward(U[mu],mu,
										 Gimpl::CovShiftIdentityBackward(U[nu], nu))));




  }
  //////////////////////////////////////////////////
  // trace of directed plaquette oriented in mu,nu plane
  //////////////////////////////////////////////////
  static void traceDirPlaquette(ComplexField &plaq,
                                const std::vector<GaugeMat> &U, const int mu,
                                const int nu) {
    GaugeMat sp(U[0].Grid());
    dirPlaquette(sp, U, mu, nu);
    plaq = trace(sp);
  }
  //////////////////////////////////////////////////
  // sum over all planes of plaquette
  //////////////////////////////////////////////////
  static void sitePlaquette(ComplexField &Plaq,
                            const std::vector<GaugeMat> &U) {
    ComplexField sitePlaq(U[0].Grid());
    Plaq = Zero();
    for (int mu = 1; mu < Nd; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        traceDirPlaquette(sitePlaq, U, mu, nu);
        Plaq = Plaq + sitePlaq;
      }
    }
  }
  //////////////////////////////////////////////////
  // sum over all x,y,z,t and over all planes of plaquette
  //////////////////////////////////////////////////
  static RealD sumPlaquette(const GaugeLorentz &Umu) {
    std::vector<GaugeMat> U(Nd, Umu.Grid());
    // inefficient here
    for (int mu = 0; mu < Nd; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    ComplexField Plaq(Umu.Grid());

    sitePlaquette(Plaq, U);
    auto Tp = sum(Plaq);
    auto p = TensorRemove(Tp);
    return p.real();
  }


  //////////////////////////////////////////////////
  // average over all x,y,z,t and over all planes of plaquette
  //////////////////////////////////////////////////
  static RealD avgPlaquette(const GaugeLorentz &Umu) {
    RealD sumplaq = sumPlaquette(Umu);
    double vol = Umu.Grid()->gSites();
    double faces = (1.0 * Nd * (Nd - 1)) / 2.0;
    return sumplaq / vol / faces / Nc; // Nd , Nc dependent... FIXME
  }

  //////////////////////////////////////////////////
  // plaquette average with open temporal boundaries
  //////////////////////////////////////////////////
  // Loops leaving [0,T] in the open direction are omitted; purely spatial
  // plaquettes on x0=0 and x0=T carry weights.boundary.  For periodic
  // parameters this is avgPlaquette(Umu) for any weights.  With 4D open
  // boundaries the weight sum is 3*V3*(2*Lt - 3 + 2*weights.boundary).
  static GaugeLoopAverage plaquetteAverage(const GaugeLorentz &Umu,
                                           const OpenGaugeBoundaryParameters &boundary,
                                           const GaugeLoopWeights &weights) {
    if (!boundary.open) {
      double vol = Umu.Grid()->gSites();
      double faces = (1.0 * Nd * (Nd - 1)) / 2.0;
      return {avgPlaquette(Umu), RealD(vol * faces)};
    }
    return LoopPath::loopAverage(Umu, LoopPath::plaquetteLoops(1.0), boundary, weights);
  }

  // Boundary weight 1/2, the action weight cG/2 at its tree-level value cG = 1,
  // independent of the cG in boundary: Nc*value equals openQCD's
  // avpl = plaq_wsum_dble()/npl, and weightSum equals npl = 6*(Lt-1)*V3.
  static GaugeLoopAverage treeLevelPlaquetteAverage(const GaugeLorentz &Umu,
                                                    const OpenGaugeBoundaryParameters &boundary) {
    return plaquetteAverage(Umu, boundary, GaugeLoopWeights::treeLevel());
  }

  // Boundary weight cG/2, as in the open gauge action.  For the Wilson action
  // S = beta * weightSum * (1 - value).  Equal to treeLevelPlaquetteAverage at
  // cG = 1.
  static GaugeLoopAverage actionCompatiblePlaquetteAverage(const GaugeLorentz &Umu,
                                                           const OpenGaugeBoundaryParameters &boundary) {
    return plaquetteAverage(Umu, boundary, GaugeLoopWeights::action(boundary));
  }

  // Boundary weight 1: every plaquette that fits in [0,T] counts once.
  static GaugeLoopAverage geometricPlaquetteAverage(const GaugeLorentz &Umu,
                                                    const OpenGaugeBoundaryParameters &boundary) {
    return plaquetteAverage(Umu, boundary, GaugeLoopWeights::geometric());
  }

  // Plaquette average per time slice of boundary.timeDir.  With the default
  // Split assignment and GaugeLoopWeights::treeLevel(), the plaquette
  // action density 2*Nc*weightSum[x0]*(1 - value[x0]) is openQCD's
  // plaq_action_slices() asl[x0] (for the same field; cG-independent
  // tree-level weights); with action(boundary) weights it is the slice
  // resolution of the open Wilson action density.
  static GaugeLoopSliceAverage plaquetteSliceAverage(const GaugeLorentz &Umu,
                                                     const OpenGaugeBoundaryParameters &boundary,
                                                     const GaugeLoopWeights &weights,
                                                     const GaugeLoopSliceAssignment assignment = GaugeLoopSliceAssignment::Split) {
    return LoopPath::loopSliceAverage(Umu, LoopPath::plaquetteLoops(1.0), boundary, weights, assignment);
  }

  //////////////////////////////////////////////////
  // sum over all spatial planes of plaquette
  //////////////////////////////////////////////////
  static void siteSpatialPlaquette(ComplexField &Plaq,
                            const std::vector<GaugeMat> &U) {
    ComplexField sitePlaq(U[0].Grid());
    Plaq = Zero();
    for (int mu = 1; mu < Nd-1; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        traceDirPlaquette(sitePlaq, U, mu, nu);
        Plaq = Plaq + sitePlaq;
      }
    }
  }

  ////////////////////////////////////
  // sum over all x,y,z and over all spatial planes of plaquette
  //////////////////////////////////////////////////
  static std::vector<RealD> timesliceSumSpatialPlaquette(const GaugeLorentz &Umu) {
    std::vector<GaugeMat> U(Nd, Umu.Grid());
    // inefficient here
    for (int mu = 0; mu < Nd; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    ComplexField Plaq(Umu.Grid());

    siteSpatialPlaquette(Plaq, U);
    typedef typename ComplexField::scalar_object sobj;
    std::vector<sobj> Tq;
    sliceSum(Plaq, Tq, Nd-1);

    std::vector<Real> out(Tq.size());
    for(int t=0;t<Tq.size();t++) out[t] = TensorRemove(Tq[t]).real();
    return out;
  }
  
  //////////////////////////////////////////////////
  // average over all x,y,z and over all spatial planes of plaquette
  //////////////////////////////////////////////////
  static std::vector<RealD> timesliceAvgSpatialPlaquette(const GaugeLorentz &Umu) {
    std::vector<RealD> sumplaq = timesliceSumSpatialPlaquette(Umu);
    int Lt = Umu.Grid()->FullDimensions()[Nd-1];
    GRID_ASSERT(sumplaq.size() == Lt);
    double vol = Umu.Grid()->gSites() / Lt;
    double faces = (1.0 * (Nd - 1)* (Nd - 2)) / 2.0;
    for(int t=0;t<Lt;t++)
      sumplaq[t] = sumplaq[t] / vol / faces / Nc; // Nd , Nc dependent... FIXME
    return sumplaq;
  }

  //////////////////////////////////////////////////
  // average Polyakov loop in mu direction over all directions != mu
  //////////////////////////////////////////////////
  static ComplexD avgPolyakovLoop(const GaugeField &Umu, const int mu) {  //assume Nd=4
    
    // Protect against bad value of mu [0, 3]
    if ((mu < 0 ) || (mu > 3)) {
      std::cout << GridLogError << "Index is not an integer inclusively between 0 and 3." << std::endl;
      exit(1);
    }

    // U_loop is U_{mu}
    GaugeMat U_loop(Umu.Grid()), P(Umu.Grid());
    ComplexD out;
    int T = Umu.Grid()->GlobalDimensions()[3];
    int X = Umu.Grid()->GlobalDimensions()[0];
    int Y = Umu.Grid()->GlobalDimensions()[1];
    int Z = Umu.Grid()->GlobalDimensions()[2];

    // Number of sites in mu direction
    int N_mu = Umu.Grid()->GlobalDimensions()[mu];

    U_loop = peekLorentz(Umu, mu); //Select direction
    P = U_loop;
    for (int t=1;t<N_mu;t++){ 
      P = Gimpl::CovShiftForward(U_loop,mu,P);
    }
   RealD norm = 1.0/(Nc*X*Y*Z*T);
   out = sum(trace(P))*norm;
   return out;   
  }  

  /////////////////////////////////////////////////
  // overload for temporal Polyakov loop
  /////////////////////////////////////////////////
  static ComplexD avgPolyakovLoop(const GaugeField &Umu) { 
    return avgPolyakovLoop(Umu, 3);
  }

  /////////////////////////////////////////////////
  // Polyakov loop with open temporal boundaries: a loop winding through the
  // open direction is not a closed loop, so it is reported as unavailable
  // (returns false, result untouched).  Other directions are unchanged.
  /////////////////////////////////////////////////
  static bool tryAvgPolyakovLoop(ComplexD &result, const GaugeField &Umu, const int mu,
                                 const OpenGaugeBoundaryParameters &boundary) {
    if (boundary.open && (mu == boundary.timeDir)) return false;
    result = avgPolyakovLoop(Umu, mu);
    return true;
  }

  //////////////////////////////////////////////////
  // average over traced single links
  //////////////////////////////////////////////////
  static RealD linkTrace(const GaugeLorentz &Umu) {
    std::vector<GaugeMat> U(Nd, Umu.Grid());

    ComplexField Tr(Umu.Grid());
    Tr = Zero();
    for (int mu = 0; mu < Nd; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
      Tr = Tr + trace(U[mu]);
    }

    auto Tp = sum(Tr);
    auto p = TensorRemove(Tp);

    double vol = Umu.Grid()->gSites();

    return p.real() / vol / (4.0 * Nc ) ;
  };

  //////////////////////////////////////////////////
  // the sum over all staples on each site in direction mu,nu
  //////////////////////////////////////////////////
  static void Staple(GaugeMat &staple, const GaugeLorentz &Umu, int mu,
                     int nu) {

    GridBase *grid = Umu.Grid();

    std::vector<GaugeMat> U(Nd, grid);
    for (int d = 0; d < Nd; d++) {
      U[d] = PeekIndex<LorentzIndex>(Umu, d);
    }
    staple = Zero();

    if (nu != mu) {

      // mu
      // ^
      // |__>  nu

      //    __
      //      |
      //    __|
      //

      staple += Gimpl::ShiftStaple(
				   Gimpl::CovShiftForward(
							  U[nu], nu,
							  Gimpl::CovShiftBackward(
										  U[mu], mu, Gimpl::CovShiftIdentityBackward(U[nu], nu))),
				   mu);

      //  __
      // |
      // |__
      //
      //
      staple += Gimpl::ShiftStaple(
				   Gimpl::CovShiftBackward(U[nu], nu,
							   Gimpl::CovShiftBackward(U[mu], mu, U[nu])),
				   mu);
    }
  }


  // For the force term
/*
  static void StapleMult(GaugeMat &staple, const GaugeLorentz &Umu, int mu) {
    GridBase *grid = Umu.Grid();
    std::vector<GaugeMat> U(Nd, grid);
    for (int d = 0; d < Nd; d++) {
      // this operation is taking too much time
      U[d] = PeekIndex<LorentzIndex>(Umu, d);
    }
    staple = Zero();
    GaugeMat tmp1(grid);
    GaugeMat tmp2(grid);

    for (int nu = 0; nu < Nd; nu++) {
      if (nu != mu) {
        // this is ~10% faster than the Staple  -- PAB: so what it gives the WRONG answers for other BC's!
        tmp1 = Cshift(U[nu], mu, 1);
        tmp2 = Cshift(U[mu], nu, 1);
        staple += tmp1* adj(U[nu]*tmp2);
        tmp2 = adj(U[mu]*tmp1)*U[nu];
        staple += Cshift(tmp2, nu, -1);
      }
    }
    staple = U[mu]*staple;
  }
*/
  //////////////////////////////////////////////////
  // the sum over all nu-oriented staples for nu != mu on each site
  //////////////////////////////////////////////////
  static void Staple(GaugeMat &staple, const GaugeLorentz &U, int mu) {

    std::vector<GaugeMat> Umu(Nd, U.Grid());
    for (int d = 0; d < Nd; d++) {
      Umu[d] = PeekIndex<LorentzIndex>(U, d);
    }
    Staple(staple, Umu, mu);
  }

  static void Staple(GaugeMat &staple, const std::vector<GaugeMat> &Umu, int mu) {

    autoView(staple_v, staple, AcceleratorWrite);
    accelerator_for(i, staple.Grid()->oSites(), Simd::Nsimd(), {
        staple_v[i] = Zero();
    });

    for (int nu = 0; nu < Nd; nu++) {

      if (nu != mu) {

        // mu
        // ^
        // |__>  nu

        //    __
        //      |
        //    __|
        //

        staple += Gimpl::ShiftStaple(
				     Gimpl::CovShiftForward(
							    Umu[nu], nu,
							    Gimpl::CovShiftBackward(
										    Umu[mu], mu, Gimpl::CovShiftIdentityBackward(Umu[nu], nu))),
				     mu);

        //  __
        // |
        // |__
        //
        //

        staple += Gimpl::ShiftStaple(
				     Gimpl::CovShiftBackward(Umu[nu], nu,
							     Gimpl::CovShiftBackward(Umu[mu], mu, Umu[nu])), mu);
      }
    }
  }

  /////////////
  //Staples for each direction mu, summed over nu != mu
  //staple: output staples for each mu (Nd)
  //U: link array (Nd)
  /////////////
  static void StapleAll(std::vector<GaugeMat> &staple, const std::vector<GaugeMat> &U) {
    GRID_ASSERT(staple.size() == Nd); GRID_ASSERT(U.size() == Nd);
    for(int mu=0;mu<Nd;mu++) Staple(staple[mu], U, mu);
  }


  //A workspace class allowing reuse of the stencil
  class WilsonLoopPaddedStencilWorkspace{
    std::unique_ptr<GeneralLocalStencil> stencil;
    size_t nshift;

    void generateStencil(GridBase* padded_grid){
      double t0 = usecond();
      
      //Generate shift arrays
      std::vector<Coordinate> shifts = this->getShifts();
      nshift = shifts.size();
      
      double t1 = usecond();
      //Generate local stencil
      stencil.reset(new GeneralLocalStencil(padded_grid,shifts));
      double t2 = usecond();
      std::cout << GridLogPerformance << " WilsonLoopPaddedWorkspace timings: coord:" << (t1-t0)/1000 << "ms, stencil:" << (t2-t1)/1000 << "ms" << std::endl;   
    }
  public:
    //Get the stencil. If not already generated, or if generated using a different Grid than in PaddedCell, it will be created on-the-fly
    const GeneralLocalStencil & getStencil(const PaddedCell &pcell){
      GRID_ASSERT(pcell.depth >= this->paddingDepth());
      if(!stencil || stencil->Grid() != (GridBase*)pcell.grids.back() ) generateStencil((GridBase*)pcell.grids.back());
      return *stencil;
    }
    size_t Nshift() const{ return nshift; }
    
    virtual std::vector<Coordinate> getShifts() const = 0;
    virtual int paddingDepth() const = 0; //padding depth required
    
    virtual ~WilsonLoopPaddedStencilWorkspace(){}
  };

  //This workspace allows the sharing of a common PaddedCell object between multiple stencil workspaces
  class WilsonLoopPaddedWorkspace{
    std::vector<WilsonLoopPaddedStencilWorkspace*> stencil_wk;
    std::unique_ptr<PaddedCell> pcell;

    void generatePcell(GridBase* unpadded_grid){
      GRID_ASSERT(stencil_wk.size());
      int max_depth = 0;
      for(auto const &s : stencil_wk) max_depth=std::max(max_depth, s->paddingDepth());
      
      pcell.reset(new PaddedCell(max_depth, dynamic_cast<GridCartesian*>(unpadded_grid)));
    }
    
  public:
    //Add a stencil definition. This should be done before the first call to retrieve a stencil object.
    //Takes ownership of the pointer
    void addStencil(WilsonLoopPaddedStencilWorkspace *stencil){
      GRID_ASSERT(!pcell);
      stencil_wk.push_back(stencil);
    }

    const GeneralLocalStencil & getStencil(const size_t stencil_idx, GridBase* unpadded_grid){
      if(!pcell || pcell->unpadded_grid != unpadded_grid) generatePcell(unpadded_grid);
      return stencil_wk[stencil_idx]->getStencil(*pcell);
    }      
    const PaddedCell & getPaddedCell(GridBase* unpadded_grid){
      if(!pcell || pcell->unpadded_grid != unpadded_grid) generatePcell(unpadded_grid);
      return *pcell;
    }
    
    ~WilsonLoopPaddedWorkspace(){
      for(auto &s : stencil_wk) delete s;
    }
  };

  //A workspace class allowing reuse of the stencil
  class StaplePaddedAllWorkspace: public WilsonLoopPaddedStencilWorkspace{
  public:
    std::vector<Coordinate> getShifts() const override{
      std::vector<Coordinate> shifts;
      for(int mu=0;mu<Nd;mu++){
	for(int nu=0;nu<Nd;nu++){
	  if(nu != mu){
	    Coordinate shift_0(Nd,0);
	    Coordinate shift_mu(Nd,0); shift_mu[mu]=1;
	    Coordinate shift_nu(Nd,0); shift_nu[nu]=1;
	    Coordinate shift_mnu(Nd,0); shift_mnu[nu]=-1;
	    Coordinate shift_mnu_pmu(Nd,0); shift_mnu_pmu[nu]=-1; shift_mnu_pmu[mu]=1;
      
	    //U_nu(x+mu)U^dag_mu(x+nu) U^dag_nu(x)
	    shifts.push_back(shift_0);
	    shifts.push_back(shift_nu);
	    shifts.push_back(shift_mu);
      
	    //U_nu^dag(x-nu+mu) U_mu^dag(x-nu) U_nu(x-nu)
	    shifts.push_back(shift_mnu);
	    shifts.push_back(shift_mnu);
	    shifts.push_back(shift_mnu_pmu);
	  }
	}
      }
      return shifts;
    }

    int paddingDepth() const override{ return 1; }
  }; 

  //Padded cell implementation of the staple method for all mu, summed over nu != mu
  //staple: output staple for each mu, summed over nu != mu (Nd)
  //U_padded: the gauge link fields padded out using the PaddedCell class
  //Cell: the padded cell class
  static void StaplePaddedAll(std::vector<GaugeMat> &staple, const std::vector<GaugeMat> &U_padded, const PaddedCell &Cell) {
    StaplePaddedAllWorkspace wk;
    StaplePaddedAll(staple,U_padded,Cell,wk.getStencil(Cell));
  }
  
  //Padded cell implementation of the staple method for all mu, summed over nu != mu
  //staple: output staple for each mu, summed over nu != mu (Nd)
  //U_padded: the gauge link fields padded out using the PaddedCell class
  //Cell: the padded cell class
  //gStencil: the precomputed generalized local stencil for the staple
  static void StaplePaddedAll(std::vector<GaugeMat> &staple, const std::vector<GaugeMat> &U_padded, const PaddedCell &Cell, const GeneralLocalStencil &gStencil)
  {
    double t0 = usecond();
    GRID_ASSERT(U_padded.size() == Nd); GRID_ASSERT(staple.size() == Nd);
    GRID_ASSERT(U_padded[0].Grid() == (GridBase*)Cell.grids.back());
    GRID_ASSERT(Cell.depth >= 1);
    GridBase *ggrid = U_padded[0].Grid(); //padded cell grid

    int shift_mu_off = gStencil._npoints/Nd;
    
    //Open views to padded gauge links and keep open over mu loop
    typedef LatticeView<typename GaugeMat::vector_object> GaugeViewType;
    size_t vsize = Nd*sizeof(GaugeViewType);
    GaugeViewType* Ug_dirs_v_host = (GaugeViewType*)malloc(vsize);
    for(int i=0;i<Nd;i++) Ug_dirs_v_host[i] = U_padded[i].View(AcceleratorRead);
    GaugeViewType* Ug_dirs_v = (GaugeViewType*)acceleratorAllocDevice(vsize);
    acceleratorCopyToDevice(Ug_dirs_v_host,Ug_dirs_v,vsize);
    
    GaugeMat gStaple(ggrid);

    int outer_off = 0;
    for(int mu=0;mu<Nd;mu++){
      { //view scope
	autoView( gStaple_v , gStaple, AcceleratorWrite);
	auto gStencil_v = gStencil.View(AcceleratorRead);
	
	accelerator_for(ss, ggrid->oSites(), (size_t)ggrid->Nsimd(), {
	    decltype(coalescedRead(Ug_dirs_v[0][0])) stencil_ss;
	    stencil_ss = Zero();
	    int off = outer_off;
	    
	    for(int nu=0;nu<Nd;nu++){
	      if(nu != mu){	  
		GeneralStencilEntry const* e = gStencil_v.GetEntry(off++,ss);
		auto U0 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(off++,ss);
		auto U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(off++,ss);
		auto U2 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
      
		stencil_ss = stencil_ss + U2 * U1 * U0;

		e = gStencil_v.GetEntry(off++,ss);
		U0 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(off++,ss);
		U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(off++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));

		stencil_ss = stencil_ss + U2 * U1 * U0;
	      }
	    }
		
	    coalescedWrite(gStaple_v[ss],stencil_ss);
	  }
	  );
      } //ensure views are all closed!
      
      staple[mu] = Cell.Extract(gStaple);
      outer_off += shift_mu_off;
    }//mu loop

    for(int i=0;i<Nd;i++) Ug_dirs_v_host[i].ViewClose();
    free(Ug_dirs_v_host);
    acceleratorFreeDevice(Ug_dirs_v);
    
    double t1=usecond();
    
    std::cout << GridLogPerformance << "StaplePaddedAll timing:" << (t1-t0)/1000 << "ms" << std::endl;   
  }

   
  //////////////////////////////////////////////////
  // the sum over all staples on each site in direction mu,nu, upper part
  //////////////////////////////////////////////////
  static void StapleUpper(GaugeMat &staple, const GaugeLorentz &Umu, int mu,
                          int nu) {
    if (nu != mu) {
      GridBase *grid = Umu.Grid();

      std::vector<GaugeMat> U(Nd, grid);
      for (int d = 0; d < Nd; d++) {
        U[d] = PeekIndex<LorentzIndex>(Umu, d);// some redundant copies
      }

      // mu
      // ^
      // |__>  nu

      //    __
      //      |
      //    __|
      //

      staple = Gimpl::ShiftStaple(
				  Gimpl::CovShiftForward(
							 U[nu], nu,
							 Gimpl::CovShiftBackward(
										 U[mu], mu, Gimpl::CovShiftIdentityBackward(U[nu], nu))),
				  mu);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  // the sum over all staples on each site in direction mu,nu, lower part
  ////////////////////////////////////////////////////////////////////////
  static void StapleLower(GaugeMat &staple, const GaugeLorentz &Umu, int mu,
                          int nu) {
    if (nu != mu) {
      GridBase *grid = Umu.Grid();

      std::vector<GaugeMat> U(Nd, grid);
      for (int d = 0; d < Nd; d++) {
        U[d] = PeekIndex<LorentzIndex>(Umu, d);// some redundant copies
      }

      // mu
      // ^
      // |__>  nu

      //  __
      // |
      // |__
      //
      //
      staple = Gimpl::ShiftStaple(
				  Gimpl::CovShiftBackward(U[nu], nu,
                                  Gimpl::CovShiftBackward(U[mu], mu, U[nu])),
          mu);

    }
  }

  //////////////////////////////////////////////////////
  //  Field Strength
  //////////////////////////////////////////////////////
  static void FieldStrength(GaugeMat &FS, const GaugeLorentz &Umu, int mu, int nu){
    // Fmn +--<--+  Ut +--<--+
    //     |     |     |     |
      //  (x)+-->--+     +-->--+(x)  - h.c.
    //     |     |     |     |
    //     +--<--+     +--<--+

    GaugeMat Vup(Umu.Grid()), Vdn(Umu.Grid());
    StapleUpper(Vup, Umu, mu, nu);
    StapleLower(Vdn, Umu, mu, nu);
    GaugeMat v = Vup - Vdn;
    GaugeMat u = PeekIndex<LorentzIndex>(Umu, mu);  // some redundant copies
    GaugeMat vu = v*u;
      //FS = 0.25*Ta(u*v + Cshift(vu, mu, -1));
      FS = (u*v + Gimpl::CshiftLink(vu, mu, -1));
      FS = 0.125*(FS - adj(FS));
  }

  // Field strength with open temporal boundaries: zero on sites where some
  // 1x1 clover leaf of some plane would leave [0,T], i.e. outside 0 < x0 < T
  // (openQCD's ftensor() convention).  Unchanged for periodic boundaries.
  static void FieldStrength(GaugeMat &FS, const GaugeLorentz &Umu, int mu, int nu,
                            const OpenGaugeBoundaryParameters &boundary){
    FieldStrength(FS, Umu, mu, nu);
    if (boundary.open) FS = LoopPath::interiorMask(Umu.Grid(), boundary, 1) * FS;
  }

  // Clover action density s(x) = -sum_{mu<nu} tr(F F), with F the traceless
  // part of FieldStrength.  This is openQCD's ym_action() density
  // (1/4) sum_{mu,nu} F^a_{mu nu}(x)^2; it is zero outside 0 < x0 < T for open
  // boundaries.  Note WilsonFlow::energyDensityCloverleaf does not remove the
  // trace.
  static void CloverEnergyDensity(ComplexField &E, const GaugeLorentz &U,
                                  const OpenGaugeBoundaryParameters &boundary){
    GaugeMat F(U.Grid()), unit(U.Grid());
    unit = 1.0 / RealD(Nc);
    E = Zero();
    for (int mu = 0; mu < Nd - 1; mu++) {
      for (int nu = mu + 1; nu < Nd; nu++) {
        FieldStrength(F, U, mu, nu, boundary);
        F = F - trace(F) * unit;
        E = E - trace(F * F);
      }
    }
  }

  // Average of the clover action density over the interior sites; weightSum
  // is the number of interior sites (the full volume if periodic).
  static GaugeLoopAverage cloverEnergyDensityAverage(const GaugeLorentz &U,
                                                     const OpenGaugeBoundaryParameters &boundary){
    ComplexField E(U.Grid());
    CloverEnergyDensity(E, U, boundary);
    ComplexField mask = LoopPath::interiorMask(U.Grid(), boundary, 1);
    const RealD sites = real(TensorRemove(sum(mask)));
    GRID_ASSERT(sites > 0.0);
    return {real(TensorRemove(sum(E))) / sites, sites};
  }

  // The same per time slice: value[x0] is the mean density on interior slices
  // and 0 (with weightSum 0) on x0 = 0 and x0 = T for open boundaries.
  static GaugeLoopSliceAverage cloverEnergyDensitySliceAverage(const GaugeLorentz &U,
                                                               const OpenGaugeBoundaryParameters &boundary){
    ComplexField E(U.Grid());
    CloverEnergyDensity(E, U, boundary);
    ComplexField mask = LoopPath::interiorMask(U.Grid(), boundary, 1);
    GaugeLoopSliceAverage result;
    std::vector<RealD> numerator = LoopPath::sliceSums(E, boundary);
    result.weightSum = LoopPath::sliceSums(mask, boundary);
    result.value.resize(numerator.size());
    for (int x0 = 0; x0 < numerator.size(); x0++)
      result.value[x0] = (result.weightSum[x0] > 0.0) ? numerator[x0] / result.weightSum[x0] : 0.0;
    return result;
  }

  // 4d topological charge density (1x1 clover)
  static void TopologicalChargeDensity(ComplexField &qfield, const GaugeLorentz &U){
    GRID_ASSERT(Nd==4);
    // Bx = -iF(y,z), By = -iF(z,y), Bz = -iF(x,y)
    GaugeMat Bx(U.Grid()), By(U.Grid()), Bz(U.Grid());
    FieldStrength(Bx, U, Ydir, Zdir);
    FieldStrength(By, U, Zdir, Xdir);
    FieldStrength(Bz, U, Xdir, Ydir);

    // Ex = -iF(t,x), Ey = -iF(t,y), Ez = -iF(t,z)
    GaugeMat Ex(U.Grid()), Ey(U.Grid()), Ez(U.Grid());
    FieldStrength(Ex, U, Tdir, Xdir);
    FieldStrength(Ey, U, Tdir, Ydir);
    FieldStrength(Ez, U, Tdir, Zdir);

    double coeff = 8.0/(32.0*M_PI*M_PI);

    qfield = coeff*trace(Bx*Ex + By*Ey + Bz*Ez);
  }

  static Real TopologicalCharge(const GaugeLorentz &U){
    // 4d topological charge
    ComplexField qfield(U.Grid());
    TopologicalChargeDensity(qfield, U);
    auto Tq = sum(qfield);
    return TensorRemove(Tq).real();
  }

  // Topological charge with open temporal boundaries: the 1x1 density is
  // zero outside 0 < x0 < T, and Q is the sum of the time-slice charges.
  static std::vector<Real> TimesliceTopologicalCharge(const GaugeLorentz &U,
                                                      const OpenGaugeBoundaryParameters &boundary){
    ComplexField qfield(U.Grid());
    TopologicalChargeDensity(qfield, U);
    qfield = LoopPath::interiorMask(U.Grid(), boundary, 1) * qfield;
    std::vector<RealD> q = LoopPath::sliceSums(qfield, boundary);
    return std::vector<Real>(q.begin(), q.end());
  }

  static Real TopologicalCharge(const GaugeLorentz &U, const OpenGaugeBoundaryParameters &boundary){
    std::vector<Real> Tq = TimesliceTopologicalCharge(U, boundary);
    Real out(0);
    for(int t=0;t<Tq.size();t++) out += Tq[t];
    return out;
  }


  //Clover-leaf Wilson loop combination for arbitrary mu-extent M and nu extent N,  mu >= nu
  //cf  https://arxiv.org/pdf/hep-lat/9701012.pdf Eq 7  for 1x2 Wilson loop    
  //Clockwise ordering
  static void CloverleafMxN(GaugeMat &FS, const GaugeMat &Umu, const GaugeMat &Unu, int mu, int nu, int M, int N){  
#define Fmu(A) Gimpl::CovShiftForward(Umu, mu, A)
#define Bmu(A) Gimpl::CovShiftBackward(Umu, mu, A)
#define Fnu(A) Gimpl::CovShiftForward(Unu, nu, A)
#define Bnu(A) Gimpl::CovShiftBackward(Unu, nu, A)
#define FmuI Gimpl::CovShiftIdentityForward(Umu, mu)
#define BmuI Gimpl::CovShiftIdentityBackward(Umu, mu)
#define FnuI Gimpl::CovShiftIdentityForward(Unu, nu)
#define BnuI Gimpl::CovShiftIdentityBackward(Unu, nu)

    //Upper right loop
    GaugeMat tmp = BmuI;
    for(int i=1;i<M;i++)
      tmp = Bmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Bnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Fmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Fnu(tmp);
      
    FS = tmp;

    //Upper left loop
    tmp = BnuI;
    for(int j=1;j<N;j++)
      tmp = Bnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Fmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Fnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Bmu(tmp);
      
    FS = FS + tmp;

    //Lower right loop
    tmp = FnuI;
    for(int j=1;j<N;j++)
      tmp = Fnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Bmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Bnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Fmu(tmp);
      
    FS = FS + tmp;

    //Lower left loop
    tmp = FmuI;
    for(int i=1;i<M;i++)
      tmp = Fmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Fnu(tmp);
    for(int i=0;i<M;i++)
      tmp = Bmu(tmp);
    for(int j=0;j<N;j++)
      tmp = Bnu(tmp);

    FS = FS + tmp;

#undef Fmu
#undef Bmu
#undef Fnu
#undef Bnu
#undef FmuI
#undef BmuI
#undef FnuI
#undef BnuI
  }

  //Field strength from MxN Wilson loop
  //Note F_numu = - F_munu
  static void FieldStrengthMxN(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu, int M, int N){  
    GaugeMat Umu = PeekIndex<LorentzIndex>(U, mu);
    GaugeMat Unu = PeekIndex<LorentzIndex>(U, nu);
    if(M == N){
      GaugeMat F(Umu.Grid());
      CloverleafMxN(F, Umu, Unu, mu, nu, M, N);
      FS = 0.125 * ( F - adj(F) );
    }else{
      //Average over both orientations
      GaugeMat horizontal(Umu.Grid()), vertical(Umu.Grid());
      CloverleafMxN(horizontal, Umu, Unu, mu, nu, M, N);
      CloverleafMxN(vertical, Umu, Unu, mu, nu, N, M);
      FS = 0.0625 * ( horizontal - adj(horizontal) + vertical - adj(vertical) );
    }
  }

  // Field strength from MxN loops with open temporal boundaries: zero on
  // sites where some clover leaf, reaching max(M,N) links in time, would leave
  // [0,T].  Unchanged for periodic boundaries.
  static void FieldStrengthMxN(GaugeMat &FS, const GaugeLorentz &U, int mu, int nu, int M, int N,
                               const OpenGaugeBoundaryParameters &boundary){
    FieldStrengthMxN(FS, U, mu, nu, M, N);
    if (boundary.open) FS = LoopPath::interiorMask(U.Grid(), boundary, std::max(M, N)) * FS;
  }

  //Topological charge contribution from MxN Wilson loops
  //cf  https://arxiv.org/pdf/hep-lat/9701012.pdf  Eq 6
  //output is the charge by timeslice: sum over timeslices to obtain the total
  static std::vector<Real> TimesliceTopologicalChargeMxN(const GaugeLorentz &U, int M, int N){
    ComplexField fsum(U.Grid());
    TopologicalChargeDensityMxN(fsum, U, M, N);

    typedef typename ComplexField::scalar_object sobj;
    std::vector<sobj> Tq;
    sliceSum(fsum, Tq, Nd-1);

    std::vector<Real> out(Tq.size());
    for(int t=0;t<Tq.size();t++) out[t] = TensorRemove(Tq[t]).real();
    return out;
  }

  // As above with open temporal boundaries: the MxN density is zero unless
  // every clover leaf fits, i.e. outside max(M,N) <= x0 <= T - max(M,N).
  static std::vector<Real> TimesliceTopologicalChargeMxN(const GaugeLorentz &U, int M, int N,
                                                         const OpenGaugeBoundaryParameters &boundary){
    ComplexField fsum(U.Grid());
    TopologicalChargeDensityMxN(fsum, U, M, N);
    fsum = LoopPath::interiorMask(U.Grid(), boundary, std::max(M, N)) * fsum;
    std::vector<RealD> q = LoopPath::sliceSums(fsum, boundary);
    return std::vector<Real>(q.begin(), q.end());
  }

  static Real TopologicalChargeMxN(const GaugeLorentz &U, int M, int N,
                                   const OpenGaugeBoundaryParameters &boundary){
    std::vector<Real> Tq = TimesliceTopologicalChargeMxN(U, M, N, boundary);
    Real out(0);
    for(int t=0;t<Tq.size();t++) out += Tq[t];
    return out;
  }

  //Topological charge density from MxN Wilson loops
  static void TopologicalChargeDensityMxN(ComplexField &fsum, const GaugeLorentz &U, int M, int N){
    GRID_ASSERT(Nd == 4);
    std::vector<std::vector<GaugeMat*> > F(Nd,std::vector<GaugeMat*>(Nd,nullptr));
    //Note F_numu = - F_munu
    //hence we only need to loop over mu,nu,rho,sigma that aren't related by permuting mu,nu  or rho,sigma
    //Use nu > mu
    for(int mu=0;mu<Nd-1;mu++){
      for(int nu=mu+1; nu<Nd; nu++){
	F[mu][nu] = new GaugeMat(U.Grid());
	FieldStrengthMxN(*F[mu][nu], U, mu, nu, M, N);
      }
    }
    Real coeff = -1./(32 * M_PI*M_PI * M*M * N*N); //overall sign to match CPS and Grid conventions, possibly related to time direction = 3 vs 0

    static const int combs[3][4] = { {0,1,2,3}, {0,2,1,3}, {0,3,1,2} };
    static const int signs[3] = { 1, -1, 1 }; //epsilon_{mu nu rho sigma}

    fsum = Zero();
    for(int c=0;c<3;c++){
      int mu = combs[c][0], nu = combs[c][1], rho = combs[c][2], sigma = combs[c][3];
      int eps = signs[c];
      fsum = fsum + (8. * coeff * eps) * trace( (*F[mu][nu]) * (*F[rho][sigma]) ); 
    }

    for(int mu=0;mu<Nd-1;mu++)
      for(int nu=mu+1; nu<Nd; nu++)
	delete F[mu][nu];
  }
  static Real TopologicalChargeMxN(const GaugeLorentz &U, int M, int N){
    std::vector<Real> Tq = TimesliceTopologicalChargeMxN(U,M,N);
    Real out(0);
    for(int t=0;t<Tq.size();t++) out += Tq[t];
    return out;
  }

  //Generate the contributions to the 5Li topological charge from Wilson loops of the following sizes
  //Use coefficients from hep-lat/9701012
  //1x1 : c1=(19.-55.*c5)/9.
  //2x2 : c2=(1-64.*c5)/9.
  //1x2 : c3=(-64.+640.*c5)/45.
  //1x3 : c4=1./5.-2.*c5
  //3x3 : c5=1./20.
  //Output array outer index contains the loops in the above order
  //Inner index is the time coordinate
  static std::vector<std::vector<Real> > TimesliceTopologicalCharge5LiContributions(const GaugeLorentz &U){
    static const int exts[5][2] = { {1,1}, {2,2}, {1,2}, {1,3}, {3,3} };       
    std::vector<std::vector<Real> > out(5);
    for(int i=0;i<5;i++){	
      out[i] = TimesliceTopologicalChargeMxN(U,exts[i][0],exts[i][1]);
    }
    return out;
  }   

  static std::vector<Real> TopologicalCharge5LiContributions(const GaugeLorentz &U){   
    static const int exts[5][2] = { {1,1}, {2,2}, {1,2}, {1,3}, {3,3} };
    std::vector<Real> out(5);
    std::cout << GridLogMessage << "Computing topological charge" << std::endl;
    for(int i=0;i<5;i++){
      out[i] = TopologicalChargeMxN(U,exts[i][0],exts[i][1]);
      std::cout << GridLogMessage << exts[i][0] << "x" << exts[i][1] << " Wilson loop contribution " << out[i] << std::endl;
    }
    return out;
  }

  //Compute the 5Li topological charge
  static std::vector<Real> TimesliceTopologicalCharge5Li(const GaugeLorentz &U){
    std::vector<std::vector<Real> > loops = TimesliceTopologicalCharge5LiContributions(U);

    double c5=1./20.;
    double c4=1./5.-2.*c5;
    double c3=(-64.+640.*c5)/45.;
    double c2=(1-64.*c5)/9.;
    double c1=(19.-55.*c5)/9.;

    int Lt = loops[0].size();
    std::vector<Real> out(Lt,0.);
    for(int t=0;t<Lt;t++)
      out[t] += c1*loops[0][t] + c2*loops[1][t] + c3*loops[2][t] + c4*loops[3][t] + c5*loops[4][t];
    return out;
  }

  static Real TopologicalCharge5Li(const GaugeLorentz &U){
    std::vector<Real> Qt = TimesliceTopologicalCharge5Li(U);
    Real Q = 0.;
    for(int t=0;t<Qt.size();t++) Q += Qt[t];
    std::cout << GridLogMessage << "5Li Topological charge: " << Q << std::endl;
    return Q;
  }

  // 5Li charge with open temporal boundaries.  The improvement combination
  // needs all five loop shapes, so the combined density is kept only where
  // the largest (3x3) clover fits: 3 <= x0 <= T-3.
  static std::vector<Real> TimesliceTopologicalCharge5Li(const GaugeLorentz &U,
                                                         const OpenGaugeBoundaryParameters &boundary){
    static const int exts[5][2] = { {1,1}, {2,2}, {1,2}, {1,3}, {3,3} };
    double c5=1./20.;
    double c4=1./5.-2.*c5;
    double c3=(-64.+640.*c5)/45.;
    double c2=(1-64.*c5)/9.;
    double c1=(19.-55.*c5)/9.;
    const double coeffs[5] = { c1, c2, c3, c4, c5 };

    ComplexField q(U.Grid()), qi(U.Grid());
    q = Zero();
    for(int i=0;i<5;i++){
      TopologicalChargeDensityMxN(qi, U, exts[i][0], exts[i][1]);
      q = q + coeffs[i] * qi;
    }
    q = LoopPath::interiorMask(U.Grid(), boundary, 3) * q;
    std::vector<RealD> Qt = LoopPath::sliceSums(q, boundary);
    return std::vector<Real>(Qt.begin(), Qt.end());
  }

  static Real TopologicalCharge5Li(const GaugeLorentz &U, const OpenGaugeBoundaryParameters &boundary){
    std::vector<Real> Qt = TimesliceTopologicalCharge5Li(U, boundary);
    Real Q = 0.;
    for(int t=0;t<Qt.size();t++) Q += Qt[t];
    return Q;
  }




  //////////////////////////////////////////////////////
  // Similar to above for rectangle is required
  //////////////////////////////////////////////////////
  static void dirRectangle(GaugeMat &rect, const std::vector<GaugeMat> &U,
                           const int mu, const int nu) {
    rect = Gimpl::CovShiftForward(
				  U[mu], mu, Gimpl::CovShiftForward(U[mu], mu, U[nu])) * // ->->|
      adj(Gimpl::CovShiftForward(
				 U[nu], nu, Gimpl::CovShiftForward(U[mu], mu, U[mu])));
    rect = rect +
      Gimpl::CovShiftForward(
			     U[mu], mu, Gimpl::CovShiftForward(U[nu], nu, U[nu])) * // ->||
      adj(Gimpl::CovShiftForward(
				 U[nu], nu, Gimpl::CovShiftForward(U[nu], nu, U[mu])));
  }
  static void traceDirRectangle(ComplexField &rect,
                                const std::vector<GaugeMat> &U, const int mu,
                                const int nu) {
    GaugeMat sp(U[0].Grid());
    dirRectangle(sp, U, mu, nu);
    rect = trace(sp);
  }
  static void siteRectangle(ComplexField &Rect,
                            const std::vector<GaugeMat> &U) {
    ComplexField siteRect(U[0].Grid());
    Rect = Zero();
    for (int mu = 1; mu < Nd; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        traceDirRectangle(siteRect, U, mu, nu);
        Rect = Rect + siteRect;
      }
    }
  }

  //////////////////////////////////////////////////
  // sum over all x,y,z,t and over all planes of plaquette
  //////////////////////////////////////////////////
  static RealD sumRectangle(const GaugeLorentz &Umu) {
    std::vector<GaugeMat> U(Nd, Umu.Grid());

    for (int mu = 0; mu < Nd; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    ComplexField Rect(Umu.Grid());

    siteRectangle(Rect, U);

    auto Tp = sum(Rect);
    auto p = TensorRemove(Tp);
    return p.real();
  }
  //////////////////////////////////////////////////
  // average over all x,y,z,t and over all planes of plaquette
  //////////////////////////////////////////////////
  static RealD avgRectangle(const GaugeLorentz &Umu) {

    RealD sumrect = sumRectangle(Umu);

    double vol = Umu.Grid()->gSites();

    double faces = (1.0 * Nd * (Nd - 1)); // 2 distinct orientations summed

    return sumrect / vol / faces / Nc; // Nd , Nc dependent... FIXME
  }

  //////////////////////////////////////////////////
  // 2x1 rectangle average with open temporal boundaries (both orientations)
  //////////////////////////////////////////////////
  // As plaquetteAverage: rectangles leaving [0,T] are omitted and purely
  // spatial rectangles on x0=0 and x0=T carry weights.boundary.  For periodic
  // parameters this is avgRectangle(Umu).  openQCD defines no rectangle
  // diagnostic, so only the action and geometric conventions are provided.
  static GaugeLoopAverage rectangleAverage(const GaugeLorentz &Umu,
                                           const OpenGaugeBoundaryParameters &boundary,
                                           const GaugeLoopWeights &weights) {
    if (!boundary.open) {
      double vol = Umu.Grid()->gSites();
      double faces = (1.0 * Nd * (Nd - 1));
      return {avgRectangle(Umu), RealD(vol * faces)};
    }
    return LoopPath::loopAverage(Umu, LoopPath::rectangleLoops(1.0), boundary, weights);
  }

  // Boundary weight cG/2 on spatial rectangles, as in the open
  // plaquette-plus-rectangle actions (and openQCD's action0).
  static GaugeLoopAverage actionCompatibleRectangleAverage(const GaugeLorentz &Umu,
                                                           const OpenGaugeBoundaryParameters &boundary) {
    return rectangleAverage(Umu, boundary, GaugeLoopWeights::action(boundary));
  }

  static GaugeLoopAverage geometricRectangleAverage(const GaugeLorentz &Umu,
                                                    const OpenGaugeBoundaryParameters &boundary) {
    return rectangleAverage(Umu, boundary, GaugeLoopWeights::geometric());
  }

  // Rectangle average per time slice (see plaquetteSliceAverage).  With Split
  // a rectangle extending two links in time counts 1/3 on each of the three
  // slices it spans.
  static GaugeLoopSliceAverage rectangleSliceAverage(const GaugeLorentz &Umu,
                                                     const OpenGaugeBoundaryParameters &boundary,
                                                     const GaugeLoopWeights &weights,
                                                     const GaugeLoopSliceAssignment assignment = GaugeLoopSliceAssignment::Split) {
    return LoopPath::loopSliceAverage(Umu, LoopPath::rectangleLoops(1.0), boundary, weights, assignment);
  }

  //////////////////////////////////////////////////
  // the sum over all staples on each site
  //////////////////////////////////////////////////
  static void RectStapleDouble(GaugeMat &U2, const GaugeMat &U, int mu) {
    U2 = U * Gimpl::CshiftLink(U, mu, 1);
  }

  ////////////////////////////////////////////////////////////////////////////
  // Hop by two optimisation strategy. Use RectStapleDouble to obtain 'U2'
  ////////////////////////////////////////////////////////////////////////////
  static void RectStapleOptimised(GaugeMat &Stap, const std::vector<GaugeMat> &U2,
                                  const std::vector<GaugeMat> &U, int mu) {

    Stap = Zero();

    GridBase *grid = U[0].Grid();

    GaugeMat Staple2x1(grid);
    GaugeMat tmp(grid);

    for (int nu = 0; nu < Nd; nu++) {
      if (nu != mu) {

        // Up staple    ___ ___
        //             |       |
        tmp = Gimpl::CshiftLink(adj(U[nu]), nu, -1);
        tmp = adj(U2[mu]) * tmp;
        tmp = Gimpl::CshiftLink(tmp, mu, -2);

        Staple2x1 = Gimpl::CovShiftForward(U[nu], nu, tmp);

        // Down staple
        //             |___ ___|
        //
        tmp = adj(U2[mu]) * U[nu];
        Staple2x1 += Gimpl::CovShiftBackward(U[nu], nu, Gimpl::CshiftLink(tmp, mu, -2));

        //              ___ ___
        //             |    ___|
        //             |___ ___|
        //

        Stap += Gimpl::CshiftLink(Gimpl::CovShiftForward(U[mu], mu, Staple2x1), mu, 1);

        //              ___ ___
        //             |___    |
        //             |___ ___|
        //

        //  tmp= Staple2x1* Cshift(U[mu],mu,-2);
        //  Stap+= Cshift(tmp,mu,1) ;
        Stap += Gimpl::CshiftLink(Staple2x1, mu, 1) * Gimpl::CshiftLink(U[mu], mu, -1);
        ;

        //       --
        //      |  |
        //
        //      |  |

        tmp = Gimpl::CshiftLink(adj(U2[nu]), nu, -2);
        tmp = Gimpl::CovShiftBackward(U[mu], mu, tmp);
        tmp = U2[nu] * Gimpl::CshiftLink(tmp, nu, 2);
        Stap += Gimpl::CshiftLink(tmp, mu, 1);

        //      |  |
        //
        //      |  |
        //       --

        tmp = Gimpl::CovShiftBackward(U[mu], mu, U2[nu]);
        tmp = adj(U2[nu]) * tmp;
        tmp = Gimpl::CshiftLink(tmp, nu, -2);
        Stap += Gimpl::CshiftLink(tmp, mu, 1);
      }
    }
  }

  static void RectStapleUnoptimised(GaugeMat &Stap, const GaugeLorentz &Umu,
                                    int mu) {
    GridBase *grid = Umu.Grid();

    std::vector<GaugeMat> U(Nd, grid);
    for (int d = 0; d < Nd; d++) {
      U[d] = PeekIndex<LorentzIndex>(Umu, d);
    }

    Stap = Zero();

    for (int nu = 0; nu < Nd; nu++) {
      if (nu != mu) {
        //           __ ___
        //          |    __ |
        //
        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftForward(
							  U[mu], mu,
							  Gimpl::CovShiftForward(
										 U[nu], nu,
										 Gimpl::CovShiftBackward(
													 U[mu], mu,
													 Gimpl::CovShiftBackward(
																 U[mu], mu,
																 Gimpl::CovShiftIdentityBackward(U[nu], nu))))),
				   mu);

        //              __
        //          |__ __ |

        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftForward(
							  U[mu], mu,
							  Gimpl::CovShiftBackward(
										  U[nu], nu,
										  Gimpl::CovShiftBackward(
													  U[mu], mu, Gimpl::CovShiftBackward(U[mu], mu, U[nu])))),
				   mu);

        //           __
        //          |__ __ |

        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftBackward(
							   U[nu], nu,
							   Gimpl::CovShiftBackward(
										   U[mu], mu,
										   Gimpl::CovShiftBackward(
													   U[mu], mu, Gimpl::CovShiftForward(U[nu], nu, U[mu])))),
				   mu);

        //           __ ___
        //          |__    |

        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftForward(
							  U[nu], nu,
							  Gimpl::CovShiftBackward(
										  U[mu], mu,
										  Gimpl::CovShiftBackward(
													  U[mu], mu, Gimpl::CovShiftBackward(U[nu], nu, U[mu])))),
				   mu);

        //       --
        //      |  |
        //
        //      |  |

        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftForward(
							  U[nu], nu,
							  Gimpl::CovShiftForward(
										 U[nu], nu,
										 Gimpl::CovShiftBackward(
													 U[mu], mu,
													 Gimpl::CovShiftBackward(
																 U[nu], nu,
																 Gimpl::CovShiftIdentityBackward(U[nu], nu))))),
				   mu);

        //      |  |
        //
        //      |  |
        //       --

        Stap += Gimpl::ShiftStaple(
				   Gimpl::CovShiftBackward(
							   U[nu], nu,
							   Gimpl::CovShiftBackward(
										   U[nu], nu,
										   Gimpl::CovShiftBackward(
													   U[mu], mu, Gimpl::CovShiftForward(U[nu], nu, U[nu])))),
				   mu);
      }
    }
  }

  static void RectStaple(GaugeMat &Stap, const GaugeLorentz &Umu, int mu) {
    RectStapleUnoptimised(Stap, Umu, mu);
  }
  static void RectStaple(const GaugeLorentz &Umu, GaugeMat &Stap,
                         std::vector<GaugeMat> &U2, std::vector<GaugeMat> &U,
                         int mu) {
    RectStapleOptimised(Stap, U2, U, mu);
  }
  //////////////////////////////////////////////////////
  //Compute the rectangular staples for all orientations
  //Stap : Array of staples (Nd)
  //U: Gauge links in each direction (Nd)
  /////////////////////////////////////////////////////
  static void RectStapleAll(std::vector<GaugeMat> &Stap, const std::vector<GaugeMat> &U){
    GRID_ASSERT(Stap.size() == Nd); GRID_ASSERT(U.size() == Nd);
    std::vector<GaugeMat> U2(Nd,U[0].Grid());
    for(int mu=0;mu<Nd;mu++) RectStapleDouble(U2[mu], U[mu], mu);
    for(int mu=0;mu<Nd;mu++) RectStapleOptimised(Stap[mu], U2, U, mu);
  }

  //A workspace class allowing reuse of the stencil
  class RectStaplePaddedAllWorkspace: public WilsonLoopPaddedStencilWorkspace{
  public:
    std::vector<Coordinate> getShifts() const override{
      std::vector<Coordinate> shifts;
      for (int mu = 0; mu < Nd; mu++){
	for (int nu = 0; nu < Nd; nu++) {
	  if (nu != mu) {
	    auto genShift = [&](int mushift,int nushift){
	      Coordinate out(Nd,0); out[mu]=mushift; out[nu]=nushift; return out;
	    };

	    //tmp6 = tmp5(x+mu) = U_mu(x+mu)U_nu(x+2mu)U_mu^dag(x+nu+mu) U_mu^dag(x+nu) U_nu^dag(x)
	    shifts.push_back(genShift(0,0));
	    shifts.push_back(genShift(0,+1));
	    shifts.push_back(genShift(+1,+1));
	    shifts.push_back(genShift(+2,0));
	    shifts.push_back(genShift(+1,0));

	    //tmp5 = tmp4(x+mu) = U_mu(x+mu)U^dag_nu(x-nu+2mu)U^dag_mu(x-nu+mu)U^dag_mu(x-nu)U_nu(x-nu)
	    shifts.push_back(genShift(0,-1));
	    shifts.push_back(genShift(0,-1));
	    shifts.push_back(genShift(+1,-1));
	    shifts.push_back(genShift(+2,-1));
	    shifts.push_back(genShift(+1,0));

	    //tmp5 = tmp4(x+mu) = U^dag_nu(x-nu+mu)U^dag_mu(x-nu)U^dag_mu(x-mu-nu)U_nu(x-mu-nu)U_mu(x-mu)
	    shifts.push_back(genShift(-1,0));
	    shifts.push_back(genShift(-1,-1));
	    shifts.push_back(genShift(-1,-1));
	    shifts.push_back(genShift(0,-1));
	    shifts.push_back(genShift(+1,-1));

	    //tmp5 = tmp4(x+mu) = U_nu(x+mu)U_mu^dag(x+nu)U_mu^dag(x-mu+nu)U_nu^dag(x-mu)U_mu(x-mu)
	    shifts.push_back(genShift(-1,0));
	    shifts.push_back(genShift(-1,0));
	    shifts.push_back(genShift(-1,+1));
	    shifts.push_back(genShift(0,+1));
	    shifts.push_back(genShift(+1,0));

	    //tmp6 = tmp5(x+mu) = U_nu(x+mu)U_nu(x+mu+nu)U_mu^dag(x+2nu)U_nu^dag(x+nu)U_nu^dag(x)
	    shifts.push_back(genShift(0,0));
	    shifts.push_back(genShift(0,+1));
	    shifts.push_back(genShift(0,+2));
	    shifts.push_back(genShift(+1,+1));
	    shifts.push_back(genShift(+1,0));

	    //tmp5 = tmp4(x+mu) = U_nu^dag(x+mu-nu)U_nu^dag(x+mu-2nu)U_mu^dag(x-2nu)U_nu(x-2nu)U_nu(x-nu)
	    shifts.push_back(genShift(0,-1));
	    shifts.push_back(genShift(0,-2));
	    shifts.push_back(genShift(0,-2));
	    shifts.push_back(genShift(+1,-2));
	    shifts.push_back(genShift(+1,-1));
	  }
	}
      }
      return shifts;
    }

    int paddingDepth() const override{ return 2; }
  }; 

  //Padded cell implementation of the rectangular staple method for all mu, summed over nu != mu
  //staple: output staple for each mu, summed over nu != mu (Nd)
  //U_padded: the gauge link fields padded out using the PaddedCell class
  //Cell: the padded cell class
  static void RectStaplePaddedAll(std::vector<GaugeMat> &staple, const std::vector<GaugeMat> &U_padded, const PaddedCell &Cell) {
    RectStaplePaddedAllWorkspace wk;
    RectStaplePaddedAll(staple,U_padded,Cell,wk.getStencil(Cell));
  }
  
  //Padded cell implementation of the rectangular staple method for all mu, summed over nu != mu
  //staple: output staple for each mu, summed over nu != mu (Nd)
  //U_padded: the gauge link fields padded out using the PaddedCell class
  //Cell: the padded cell class
  //gStencil: the stencil
  static void RectStaplePaddedAll(std::vector<GaugeMat> &staple, const std::vector<GaugeMat> &U_padded, const PaddedCell &Cell, const GeneralLocalStencil &gStencil) {
    double t0 = usecond();
    GRID_ASSERT(U_padded.size() == Nd); GRID_ASSERT(staple.size() == Nd);
    GRID_ASSERT(U_padded[0].Grid() == (GridBase*)Cell.grids.back());
    GRID_ASSERT(Cell.depth >= 2);
    GridBase *ggrid = U_padded[0].Grid(); //padded cell grid

    size_t nshift = gStencil._npoints;
    int mu_off_delta = nshift / Nd;
    
    //Open views to padded gauge links and keep open over mu loop
    typedef LatticeView<typename GaugeMat::vector_object> GaugeViewType;
    size_t vsize = Nd*sizeof(GaugeViewType);
    GaugeViewType* Ug_dirs_v_host = (GaugeViewType*)malloc(vsize);
    for(int i=0;i<Nd;i++) Ug_dirs_v_host[i] = U_padded[i].View(AcceleratorRead);
    GaugeViewType* Ug_dirs_v = (GaugeViewType*)acceleratorAllocDevice(vsize);
    acceleratorCopyToDevice(Ug_dirs_v_host,Ug_dirs_v,vsize);

    GaugeMat gStaple(ggrid); //temp staple object on padded grid

    int offset = 0;
    for(int mu=0; mu<Nd; mu++){

      { //view scope
	autoView( gStaple_v , gStaple, AcceleratorWrite);
	auto gStencil_v = gStencil.View(AcceleratorRead);

	accelerator_for(ss, ggrid->oSites(), (size_t)ggrid->Nsimd(), {
	    decltype(coalescedRead(Ug_dirs_v[0][0])) stencil_ss;
	    stencil_ss = Zero();
	    int s=offset;
	    for(int nu=0;nu<Nd;nu++){
	      if(nu != mu){
		//tmp6 = tmp5(x+mu) = U_mu(x+mu)U_nu(x+2mu)U_mu^dag(x+nu+mu) U_mu^dag(x+nu) U_nu^dag(x)
		GeneralStencilEntry const* e = gStencil_v.GetEntry(s++,ss);
		auto U0 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		auto U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		auto U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		auto U3 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		auto U4 = coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd);
	    
		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;

		//tmp5 = tmp4(x+mu) = U_mu(x+mu)U^dag_nu(x-nu+2mu)U^dag_mu(x-nu+mu)U^dag_mu(x-nu)U_nu(x-nu)
		e = gStencil_v.GetEntry(s++,ss);
		U0 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U3 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U4 = coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd);

		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;

		//tmp5 = tmp4(x+mu) = U^dag_nu(x-nu+mu)U^dag_mu(x-nu)U^dag_mu(x-mu-nu)U_nu(x-mu-nu)U_mu(x-mu)
		e = gStencil_v.GetEntry(s++,ss);
		U0 = coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U1 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U3 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U4 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));

		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;

		//tmp5 = tmp4(x+mu) = U_nu(x+mu)U_mu^dag(x+nu)U_mu^dag(x-mu+nu)U_nu^dag(x-mu)U_mu(x-mu)
		e = gStencil_v.GetEntry(s++,ss);
		U0 = coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U3 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U4 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);

		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;

		//tmp6 = tmp5(x+mu) = U_nu(x+mu)U_nu(x+mu+nu)U_mu^dag(x+2nu)U_nu^dag(x+nu)U_nu^dag(x)
		e = gStencil_v.GetEntry(s++,ss);
		U0 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U1 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U3 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U4 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);

		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;   

		//tmp5 = tmp4(x+mu) = U_nu^dag(x+mu-nu)U_nu^dag(x+mu-2nu)U_mu^dag(x-2nu)U_nu(x-2nu)U_nu(x-nu)
		e = gStencil_v.GetEntry(s++,ss);
		U0 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U1 = coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd);
		e = gStencil_v.GetEntry(s++,ss);
		U2 = adj(coalescedReadGeneralPermute(Ug_dirs_v[mu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U3 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));
		e = gStencil_v.GetEntry(s++,ss);
		U4 = adj(coalescedReadGeneralPermute(Ug_dirs_v[nu][e->_offset], e->_permute, Nd));

		stencil_ss = stencil_ss + U4*U3*U2*U1*U0;   

	      }
	    }
	    coalescedWrite(gStaple_v[ss],stencil_ss);
	  }
	  );
	offset += mu_off_delta;
      }//kernel/view scope

      staple[mu] = Cell.Extract(gStaple);    
    }//mu loop
  
    for(int i=0;i<Nd;i++) Ug_dirs_v_host[i].ViewClose();
    free(Ug_dirs_v_host);
    acceleratorFreeDevice(Ug_dirs_v);
    
    double t1 = usecond();
    
    std::cout << GridLogPerformance << "RectStaplePaddedAll timings:" << (t1-t0)/1000 << "ms" << std::endl;   
  }

  //A workspace for reusing the PaddedCell and GeneralLocalStencil objects
  class StapleAndRectStapleAllWorkspace: public WilsonLoopPaddedWorkspace{
  public:
    StapleAndRectStapleAllWorkspace(){
      this->addStencil(new StaplePaddedAllWorkspace);
      this->addStencil(new RectStaplePaddedAllWorkspace);
    }
  };     
    
  //////////////////////////////////////////////////////
  //Compute the 1x1 and 1x2 staples for all orientations
  //Stap : Array of staples (Nd)
  //RectStap: Array of rectangular staples (Nd)
  //U: Gauge links in each direction (Nd)
  /////////////////////////////////////////////////////
  static void StapleAndRectStapleAll(std::vector<GaugeMat> &Stap, std::vector<GaugeMat> &RectStap, const std::vector<GaugeMat> &U){
    StapleAndRectStapleAllWorkspace wk;
    StapleAndRectStapleAll(Stap,RectStap,U,wk);
  }
  
  //////////////////////////////////////////////////////
  //Compute the 1x1 and 1x2 staples for all orientations
  //Stap : Array of staples (Nd)
  //RectStap: Array of rectangular staples (Nd)
  //U: Gauge links in each direction (Nd)
  //wk: a workspace containing stored PaddedCell and GeneralLocalStencil objects to maximize reuse
  /////////////////////////////////////////////////////
  static void StapleAndRectStapleAll(std::vector<GaugeMat> &Stap, std::vector<GaugeMat> &RectStap, const std::vector<GaugeMat> &U, StapleAndRectStapleAllWorkspace &wk){
#if 0
    StapleAll(Stap, U);
    RectStapleAll(RectStap, U);
#else
    double t0 = usecond();

    GridCartesian* unpadded_grid = dynamic_cast<GridCartesian*>(U[0].Grid());
    const PaddedCell &Ghost = wk.getPaddedCell(unpadded_grid);
        
    CshiftImplGauge<Gimpl> cshift_impl;
    std::vector<GaugeMat> U_pad(Nd, Ghost.grids.back());
    for(int mu=0;mu<Nd;mu++) U_pad[mu] = Ghost.Exchange(U[mu], cshift_impl);
    double t1 = usecond();
    StaplePaddedAll(Stap, U_pad, Ghost, wk.getStencil(0,unpadded_grid) );
    double t2 = usecond();
    RectStaplePaddedAll(RectStap, U_pad, Ghost, wk.getStencil(1,unpadded_grid));
    double t3 = usecond();
    std::cout << GridLogPerformance << "StapleAndRectStapleAll timings: pad:" << (t1-t0)/1000 << "ms, staple:" << (t2-t1)/1000 << "ms, rect-staple:" << (t3-t2)/1000 << "ms" << std::endl;
#endif
  }

  //////////////////////////////////////////////////
  // Wilson loop of size (R1, R2), oriented in mu,nu plane
  //////////////////////////////////////////////////
  static void wilsonLoop(GaugeMat &wl, const std::vector<GaugeMat> &U,
                           const int Rmu, const int Rnu,
                           const int mu, const int nu) {
    wl = U[nu];

    for(int i = 0; i < Rnu-1; i++){
      wl = Gimpl::CovShiftForward(U[nu], nu, wl);
    }

    for(int i = 0; i < Rmu; i++){
      wl = Gimpl::CovShiftForward(U[mu], mu, wl);
    }

    for(int i = 0; i < Rnu; i++){
      wl = Gimpl::CovShiftBackward(U[nu], nu, wl);
    }

    for(int i = 0; i < Rmu; i++){
      wl = Gimpl::CovShiftBackward(U[mu], mu, wl);
    }
  }
  //////////////////////////////////////////////////
  // trace of Wilson Loop oriented in mu,nu plane
  //////////////////////////////////////////////////
  static void traceWilsonLoop(LatticeComplex &wl,
                                const std::vector<GaugeMat> &U,
                                const int Rmu, const int Rnu,
                                const int mu, const int nu) {
    GaugeMat sp(U[0].Grid());
    wilsonLoop(sp, U, Rmu, Rnu, mu, nu);
    wl = trace(sp);
  }
  //////////////////////////////////////////////////
  // sum over all planes of Wilson loop
  //////////////////////////////////////////////////
  static void siteWilsonLoop(LatticeComplex &Wl,
                            const std::vector<GaugeMat> &U,
                            const int R1, const int R2) {
    LatticeComplex siteWl(U[0].Grid());
    Wl = Zero();
    for (int mu = 1; mu < U[0].Grid()->_ndimension; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        traceWilsonLoop(siteWl, U, R1, R2, mu, nu);
        Wl = Wl + siteWl;
        traceWilsonLoop(siteWl, U, R2, R1, mu, nu);
        Wl = Wl + siteWl;
      }
    }
  }
  //////////////////////////////////////////////////
  // sum over planes of Wilson loop with length R1
  // in the time direction
  //////////////////////////////////////////////////
  static void siteTimelikeWilsonLoop(LatticeComplex &Wl,
                            const std::vector<GaugeMat> &U,
                            const int R1, const int R2) {
    LatticeComplex siteWl(U[0].Grid());

    int ndim = U[0].Grid()->_ndimension;

    Wl = Zero();
    for (int nu = 0; nu < ndim - 1; nu++) {
      traceWilsonLoop(siteWl, U, R1, R2, ndim-1, nu);
      Wl = Wl + siteWl;
    }
  }
  //////////////////////////////////////////////////
  // sum Wilson loop over all planes orthogonal to the time direction
  //////////////////////////////////////////////////
  static void siteSpatialWilsonLoop(LatticeComplex &Wl,
                            const std::vector<GaugeMat> &U,
                            const int R1, const int R2) {
    LatticeComplex siteWl(U[0].Grid());

    Wl = Zero();
    for (int mu = 1; mu < U[0].Grid()->_ndimension - 1; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        traceWilsonLoop(siteWl, U, R1, R2, mu, nu);
        Wl = Wl + siteWl;
        traceWilsonLoop(siteWl, U, R2, R1, mu, nu);
        Wl = Wl + siteWl;
      }
    }
  }
  //////////////////////////////////////////////////
  // sum over all x,y,z,t and over all planes of Wilson loop
  //////////////////////////////////////////////////
  static Real sumWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    std::vector<GaugeMat> U(4, Umu.Grid());

    for (int mu = 0; mu < Umu.Grid()->_ndimension; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    LatticeComplex Wl(Umu.Grid());

    siteWilsonLoop(Wl, U, R1, R2);

    TComplex Tp = sum(Wl);
    Complex p = TensorRemove(Tp);
    return p.real();
  }
  //////////////////////////////////////////////////
  // sum over all x,y,z,t and over all planes of timelike Wilson loop
  //////////////////////////////////////////////////
  static Real sumTimelikeWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    std::vector<GaugeMat> U(4, Umu.Grid());

    for (int mu = 0; mu < Umu.Grid()->_ndimension; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    LatticeComplex Wl(Umu.Grid());

    siteTimelikeWilsonLoop(Wl, U, R1, R2);

    TComplex Tp = sum(Wl);
    Complex p = TensorRemove(Tp);
    return p.real();
  }
  //////////////////////////////////////////////////
  // sum over all x,y,z,t and over all planes of spatial Wilson loop
  //////////////////////////////////////////////////
  static Real sumSpatialWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    std::vector<GaugeMat> U(4, Umu.Grid());

    for (int mu = 0; mu < Umu.Grid()->_ndimension; mu++) {
      U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    LatticeComplex Wl(Umu.Grid());

    siteSpatialWilsonLoop(Wl, U, R1, R2);

    TComplex Tp = sum(Wl);
    Complex p = TensorRemove(Tp);
    return p.real();
  }
  //////////////////////////////////////////////////
  // average over all x,y,z,t and over all planes of Wilson loop
  //////////////////////////////////////////////////
  static Real avgWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    int ndim = Umu.Grid()->_ndimension;
    Real sumWl = sumWilsonLoop(Umu, R1, R2);
    Real vol = Umu.Grid()->gSites();
    Real faces = 1.0 * ndim * (ndim - 1);
    return sumWl / vol / faces / Nc; // Nc dependent... FIXME
  }
  //////////////////////////////////////////////////
  // average over all x,y,z,t and over all planes of timelike Wilson loop
  //////////////////////////////////////////////////
  static Real avgTimelikeWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    int ndim = Umu.Grid()->_ndimension;
    Real sumWl = sumTimelikeWilsonLoop(Umu, R1, R2);
    Real vol = Umu.Grid()->gSites();
    Real faces = 1.0 * (ndim - 1);
    return sumWl / vol / faces / Nc; // Nc dependent... FIXME
  }
  //////////////////////////////////////////////////
  // average over all x,y,z,t and over all planes of spatial Wilson loop
  //////////////////////////////////////////////////
  static Real avgSpatialWilsonLoop(const GaugeLorentz &Umu,
                            const int R1, const int R2) {
    int ndim = Umu.Grid()->_ndimension;
    Real sumWl = sumSpatialWilsonLoop(Umu, R1, R2);
    Real vol = Umu.Grid()->gSites();
    Real faces = 1.0 * (ndim - 1) * (ndim - 2);
    return sumWl / vol / faces / Nc; // Nc dependent... FIXME
  }
};

typedef WilsonLoops<PeriodicGimplR> ColourWilsonLoops;
typedef WilsonLoops<PeriodicGimplR> U1WilsonLoops;
typedef WilsonLoops<PeriodicGimplR> SU2WilsonLoops;
typedef WilsonLoops<PeriodicGimplR> SU3WilsonLoops;

NAMESPACE_END(Grid);

#endif
