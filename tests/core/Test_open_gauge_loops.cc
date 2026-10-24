/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_open_gauge_loops.cc

    Copyright (C) 2026

    Author: Matthew Black <matthewkblack@protonmail.com>

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
#include <Grid/Grid.h>

using namespace Grid;

typedef WilsonLoops<PeriodicGimplR> Loops;

static bool nearlyEqual(RealD a, RealD b, RealD tol = 1.0e-12)
{
  return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

static void check(bool ok, const std::string &what)
{
  std::cout << GridLogMessage << (ok ? "pass: " : "FAIL: ") << what << std::endl;
  GRID_ASSERT(ok);
}

static void logAverage(const std::string &label, const GaugeLoopAverage &a)
{
  std::cout << GridLogMessage << label << " value " << a.value << ", weightSum " << a.weightSum << std::endl;
}

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  const int t = Nd - 1;
  const int Lt = latt_size[t];
  GRID_ASSERT(Lt >= 5);                               // need sufficient sites for 1x2 topological charge
  const RealD V = grid.gSites();
  const RealD V3 = V / Lt;
  const RealD ntime = Nd - 1;                         // time-like planes
  const RealD nspace = (Nd - 1) * (Nd - 2) / 2.0;     // space-like planes

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({45, 12, 81, 9});

  LatticeGaugeField U(&grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  const RealD cG = 1.4;
  OpenGaugeBoundaryParameters open = OpenGaugeBoundaryParameters::Open(t, cG);
  OpenGaugeBoundaryParameters open1 = OpenGaugeBoundaryParameters::Open(t, 1.0);
  // cG has no meaning for periodic boundaries; use a non-default value to show it is ignored.
  OpenGaugeBoundaryParameters periodic(false, t, cG);

  const GaugeLoopWeights weights[3] = {GaugeLoopWeights::geometric(),
                                       GaugeLoopWeights::treeLevel(),
                                       GaugeLoopWeights::action(open)};

  //////////////////////////////////////////////////////////////////////////////
  // 1. Periodic parameters reproduce the legacy observables for any weighting,
  //    both through the fast path and through the generic loop evaluator.
  //////////////////////////////////////////////////////////////////////////////
  const RealD plaq = Loops::avgPlaquette(U);
  const RealD rect = Loops::avgRectangle(U);
  for (const auto &w : weights) {
    GaugeLoopAverage p = Loops::plaquetteAverage(U, periodic, w);
    GaugeLoopAverage r = Loops::rectangleAverage(U, periodic, w);
    check(nearlyEqual(p.value, plaq) && nearlyEqual(p.weightSum, V * Nd * (Nd - 1) / 2.0), "periodic plaquette average = avgPlaquette");
    check(nearlyEqual(r.value, rect) && nearlyEqual(r.weightSum, V * Nd * (Nd - 1)), "periodic rectangle average = avgRectangle");

    GaugeLoopAverage pg = Loops::LoopPath::loopAverage(U, Loops::LoopPath::plaquetteLoops(1.0), periodic, w);
    GaugeLoopAverage rg = Loops::LoopPath::loopAverage(U, Loops::LoopPath::rectangleLoops(1.0), periodic, w);
    check(nearlyEqual(pg.value, plaq, 1.0e-10) && nearlyEqual(pg.weightSum, p.weightSum), "generic periodic plaquette = avgPlaquette");
    check(nearlyEqual(rg.value, rect, 1.0e-10) && nearlyEqual(rg.weightSum, r.weightSum), "generic periodic rectangle = avgRectangle");
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2. Open weight sums: hand counts for each boundary weight w.
  //    plaquettes: ntime*V3*(Lt-1) + nspace*V3*(Lt-2+2w)
  //    rectangles: ntime*V3*((Lt-2)+(Lt-1)) + 2*nspace*V3*(Lt-2+2w)
  //    For Nd=4 and w=1/2 the plaquette sum is openQCD's npl = 6*(Lt-1)*V3.
  //////////////////////////////////////////////////////////////////////////////
  for (const auto &w : weights) {
    const RealD b = w.boundary;
    const RealD plaqWeight = ntime * V3 * (Lt - 1) + nspace * V3 * (Lt - 2 + 2 * b);
    const RealD rectWeight = ntime * V3 * (2 * Lt - 3) + 2 * nspace * V3 * (Lt - 2 + 2 * b);
    check(nearlyEqual(Loops::plaquetteAverage(U, open, w).weightSum, plaqWeight), "open plaquette weight sum");
    check(nearlyEqual(Loops::rectangleAverage(U, open, w).weightSum, rectWeight), "open rectangle weight sum");
  }
  if (Nd == 4) {
    check(nearlyEqual(Loops::treeLevelPlaquetteAverage(U, open).weightSum, 6.0 * (Lt - 1) * V3), "openQCD npl = 6(Lt-1)V3");
    check(nearlyEqual(Loops::actionCompatiblePlaquetteAverage(U, open).weightSum, 3.0 * V3 * (2 * Lt - 3 + cG)), "action weight sum = 3V3(2Lt-3+cG)");
    check(nearlyEqual(Loops::geometricPlaquetteAverage(U, open).weightSum, 3.0 * V3 * (2 * Lt - 1)), "geometric weight sum = 3V3(2Lt-1)");
  }

  //////////////////////////////////////////////////////////////////////////////
  // 3. cG enters only the action convention; at cG=1 it equals the tree-level one.
  //////////////////////////////////////////////////////////////////////////////
  {
    OpenGaugeBoundaryParameters other = OpenGaugeBoundaryParameters::Open(t, 0.6);
    check(nearlyEqual(Loops::treeLevelPlaquetteAverage(U, open).value, Loops::treeLevelPlaquetteAverage(U, other).value, 1.0e-14), "tree-level plaquette independent of cG");
    check(nearlyEqual(Loops::geometricPlaquetteAverage(U, open).value, Loops::geometricPlaquetteAverage(U, other).value, 1.0e-14), "geometric plaquette independent of cG");
    check(nearlyEqual(Loops::geometricRectangleAverage(U, open).value, Loops::geometricRectangleAverage(U, other).value, 1.0e-14), "geometric rectangle independent of cG");

    GaugeLoopAverage a1 = Loops::actionCompatiblePlaquetteAverage(U, open1);
    GaugeLoopAverage q1 = Loops::treeLevelPlaquetteAverage(U, open1);
    check(nearlyEqual(a1.value, q1.value, 1.0e-14) && nearlyEqual(a1.weightSum, q1.weightSum, 1.0e-14), "action plaquette = tree-level plaquette at cG=1");

    logAverage("tree-level plaquette (cG=1.4)", Loops::treeLevelPlaquetteAverage(U, open));
    logAverage("action plaquette     (cG=1.4)", Loops::actionCompatiblePlaquetteAverage(U, open));
    logAverage("geometric plaquette          ", Loops::geometricPlaquetteAverage(U, open));
    logAverage("action rectangle     (cG=1.4)", Loops::actionCompatibleRectangleAverage(U, open));
    logAverage("geometric rectangle          ", Loops::geometricRectangleAverage(U, open));
  }

  //////////////////////////////////////////////////////////////////////////////
  // 4. Action identities: S = sum_loops c * W * (1 - average).
  //////////////////////////////////////////////////////////////////////////////
  {
    const RealD beta = 1.7;
    WilsonGaugeAction<PeriodicGimplR> wilson(beta, open);
    GaugeLoopAverage p = Loops::actionCompatiblePlaquetteAverage(U, open);
    const RealD S = wilson.S(U);
    check(nearlyEqual(S, beta * p.weightSum * (1.0 - p.value), 1.0e-10), "open Wilson S = beta W (1 - P)");

    const RealD c_plaq = 2.3, c_rect = -0.17;
    PlaqPlusRectangleAction<PeriodicGimplR> improved(c_plaq, c_rect, open);
    GaugeLoopAverage r = Loops::actionCompatibleRectangleAverage(U, open);
    const RealD Si = improved.S(U);
    check(nearlyEqual(Si, c_plaq * p.weightSum * (1.0 - p.value) + c_rect * r.weightSum * (1.0 - r.value), 1.0e-10), "open plaquette+rectangle S = c_p W_p (1 - P) + c_r W_r (1 - R)");
  }

  //////////////////////////////////////////////////////////////////////////////
  // 5. Unit gauge field: every average is 1.
  //////////////////////////////////////////////////////////////////////////////
  {
    LatticeGaugeField Uc(&grid);
    SU<Nc>::ColdConfiguration(pRNG, Uc);
    for (const auto &w : weights) {
      check(nearlyEqual(Loops::plaquetteAverage(Uc, open, w).value, 1.0), "unit gauge plaquette = 1");
      check(nearlyEqual(Loops::rectangleAverage(Uc, open, w).value, 1.0), "unit gauge rectangle = 1");
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 6. No wrap-around: changing the unused forward temporal links at x0=T
  //    leaves every open observable unchanged (but not the periodic one).
  //////////////////////////////////////////////////////////////////////////////
  {
    LatticeGaugeField W(&grid), Uw(&grid);
    SU<Nc>::HotConfiguration(pRNG, W);
    LatticeColourMatrix Ut = PeekIndex<LorentzIndex>(U, t);
    LatticeColourMatrix Wt = PeekIndex<LorentzIndex>(W, t);
    LatticeInteger x0(&grid);
    LatticeCoordinate(x0, t);
    Ut = where(x0 == Integer(Lt - 1), Wt, Ut);
    Uw = U;
    PokeIndex<LorentzIndex>(Uw, Ut, t);

    for (const auto &w : weights) {
      check(nearlyEqual(Loops::plaquetteAverage(U, open, w).value, Loops::plaquetteAverage(Uw, open, w).value, 1.0e-13), "open plaquette ignores the x0=T temporal link");
      check(nearlyEqual(Loops::rectangleAverage(U, open, w).value, Loops::rectangleAverage(Uw, open, w).value, 1.0e-13), "open rectangle ignores the x0=T temporal link");
    }
    check(!nearlyEqual(Loops::avgPlaquette(U), Loops::avgPlaquette(Uw), 1.0e-8), "periodic plaquette does see the x0=T temporal link");
  }

  //////////////////////////////////////////////////////////////////////////////
  // 7. Polyakov loops: unavailable along the open direction only.
  //////////////////////////////////////////////////////////////////////////////
  if (Nd == 4) {
    const ComplexD dummy(-7.0, 3.0);
    ComplexD result = dummy;
    check(!Loops::tryAvgPolyakovLoop(result, U, t, open) && (result == dummy), "open temporal Polyakov loop unavailable");
    check(Loops::tryAvgPolyakovLoop(result, U, t, periodic) && (result == Loops::avgPolyakovLoop(U, t)), "periodic temporal Polyakov loop available");
    for (int mu = 0; mu < Nd - 1; ++mu) {
      check(Loops::tryAvgPolyakovLoop(result, U, mu, open) && (result == Loops::avgPolyakovLoop(U, mu)), "open spatial Polyakov loop unchanged");
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 8. Time-slice profiles of loop averages.
  //    The slices add up to the global average for every weighting and both assignments
  //    Hand counts of the slice weights (per V3):
  //      Split : x0 = 0, T  -> nspace*w + ntime/2;  otherwise nspace + ntime
  //      Origin: x0 = 0     -> nspace*w + ntime;   0 < x0 < T -> nspace + ntime;
  //              x0 = T     -> nspace*w
  //////////////////////////////////////////////////////////////////////////////
  {
    const GaugeLoopSliceAssignment assignments[2] = {GaugeLoopSliceAssignment::Split, GaugeLoopSliceAssignment::Origin};
    const OpenGaugeBoundaryParameters bcs[2] = {open, periodic};
    for (const auto &bc : bcs) {
      for (const auto &w : weights) {
        for (const auto a : assignments) {
          const GaugeLoopAverage p = Loops::plaquetteAverage(U, bc, w);
          const GaugeLoopAverage r = Loops::rectangleAverage(U, bc, w);
          const GaugeLoopSliceAverage ps = Loops::plaquetteSliceAverage(U, bc, w, a);
          const GaugeLoopSliceAverage rs = Loops::rectangleSliceAverage(U, bc, w, a);
          check(ps.value.size() == Lt && rs.value.size() == Lt, "one entry per time slice");
          RealD pW = 0, pN = 0, rW = 0, rN = 0;
          for (int x0 = 0; x0 < Lt; ++x0) {
            pW += ps.weightSum[x0]; pN += ps.weightSum[x0] * ps.value[x0];
            rW += rs.weightSum[x0]; rN += rs.weightSum[x0] * rs.value[x0];
          }
          check(nearlyEqual(pW, p.weightSum) && nearlyEqual(pN, p.weightSum * p.value, 1.0e-10), "plaquette slices add up to the global average");
          check(nearlyEqual(rW, r.weightSum) && nearlyEqual(rN, r.weightSum * r.value, 1.0e-10), "rectangle slices add up to the global average");

          if (bc.open) {
            const RealD b = w.boundary;
            for (int x0 = 0; x0 < Lt; ++x0) {
              const bool edge = (x0 == 0) || (x0 == Lt - 1);
              RealD expected;
              if (a == GaugeLoopSliceAssignment::Split)
                expected = edge ? nspace * b + 0.5 * ntime : nspace + ntime;
              else
                expected = (x0 == Lt - 1) ? nspace * b : (edge ? nspace * b : nspace) + ntime;
              check(nearlyEqual(ps.weightSum[x0], V3 * expected), "plaquette slice weight hand count");
            }
          } else {
            for (int x0 = 0; x0 < Lt; ++x0)
              check(nearlyEqual(ps.weightSum[x0], V3 * Nd * (Nd - 1) / 2.0), "periodic plaquette slice weight");
          }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 8b. Cross-check against the legacy timesliceAvgSpatialPlaquette.
  //     Space-like plaquettes lie on one slice, so Split and Origin agree, and
  //     all of them on a given slice carry the same weight.  Hence the slice
  //     values equal the legacy ones on every slice, for periodic and open
  //     boundaries and every weighting; only the weight sum of the two
  //     boundary slices changes, to V3*nspace*w.
  //////////////////////////////////////////////////////////////////////////////
  {
    std::vector<Loops::LoopPath::Loop> spatialLoops;
    for (const auto &loop : Loops::LoopPath::plaquetteLoops(1.0)) {
      if (Loops::LoopPath::isSpatial(loop.path, t)) spatialLoops.push_back(loop);
    }
    check(spatialLoops.size() == nspace, "space-like plaquette orientations");

    const std::vector<RealD> legacy = Loops::timesliceAvgSpatialPlaquette(U);
    const GaugeLoopSliceAssignment assignments[2] = {GaugeLoopSliceAssignment::Split, GaugeLoopSliceAssignment::Origin};
    const OpenGaugeBoundaryParameters bcs[2] = {periodic, open};
    for (const auto &bc : bcs) {
      for (const auto &w : weights) {
        for (const auto a : assignments) {
          const GaugeLoopSliceAverage s = Loops::LoopPath::loopSliceAverage(U, spatialLoops, bc, w, a);
          for (int x0 = 0; x0 < Lt; ++x0) {
            const bool edge = bc.open && ((x0 == 0) || (x0 == Lt - 1));
            check(nearlyEqual(s.value[x0], legacy[x0], 1.0e-10), "spatial plaquette slice = timesliceAvgSpatialPlaquette");
            check(nearlyEqual(s.weightSum[x0], V3 * nspace * (edge ? w.boundary : 1.0)), "spatial plaquette slice weight");
          }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 9. Field strength, clover action density and topological charge.
  //    Interior sites see exactly the periodic values (no wrap-around link is
  //    involved); sites within the clover extent of a boundary are zero.
  //////////////////////////////////////////////////////////////////////////////
  if (Nd == 4) {
    // Periodic parameters reproduce the legacy charges.
    check(nearlyEqual(Loops::TopologicalCharge(U, periodic), Loops::TopologicalCharge(U), 1.0e-10), "periodic Q = legacy Q");
    check(nearlyEqual(Loops::TopologicalChargeMxN(U, 1, 2, periodic), Loops::TopologicalChargeMxN(U, 1, 2), 1.0e-10), "periodic Q(1x2) = legacy");

    // Legacy 1x1 slices, as reference for the interior.
    Loops::ComplexField qfield(&grid);
    Loops::TopologicalChargeDensity(qfield, U);
    const std::vector<RealD> q1 = Loops::LoopPath::sliceSums(qfield, periodic);
    const std::vector<Real> q12 = Loops::TimesliceTopologicalChargeMxN(U, 1, 2);

    const std::vector<Real> o1 = Loops::TimesliceTopologicalCharge(U, open);
    const std::vector<Real> o12 = Loops::TimesliceTopologicalChargeMxN(U, 1, 2, open);
    const GaugeLoopSliceAverage Ep = Loops::cloverEnergyDensitySliceAverage(U, periodic);
    const GaugeLoopSliceAverage Eo = Loops::cloverEnergyDensitySliceAverage(U, open);
    RealD Qsum = 0, Esum = 0, Wsum = 0;
    for (int x0 = 0; x0 < Lt; ++x0) {
      const bool in1 = (x0 >= 1) && (x0 <= Lt - 2);
      const bool in2 = (x0 >= 2) && (x0 <= Lt - 3);
      check(in1 ? nearlyEqual(o1[x0], q1[x0], 1.0e-10) : (o1[x0] == 0.0), "open Q slice: periodic value inside 0<x0<T, zero outside");
      check(in2 ? nearlyEqual(o12[x0], q12[x0], 1.0e-10) : (o12[x0] == 0.0), "open Q(1x2) slice: periodic value inside 2<=x0<=T-2, zero outside");
      check(in1 ? (nearlyEqual(Eo.value[x0], Ep.value[x0], 1.0e-10) && nearlyEqual(Eo.weightSum[x0], V3))
            : (Eo.value[x0] == 0.0 && Eo.weightSum[x0] == 0.0), "open clover density slice");
      Qsum += o1[x0]; Esum += Eo.value[x0] * Eo.weightSum[x0]; Wsum += Eo.weightSum[x0];
    }
    check(nearlyEqual(Qsum, Loops::TopologicalCharge(U, open), 1.0e-12), "open Q = sum of slices");
    const GaugeLoopAverage Eavg = Loops::cloverEnergyDensityAverage(U, open);
    check(nearlyEqual(Eavg.weightSum, V3 * (Lt - 2)) && nearlyEqual(Wsum, Eavg.weightSum) && nearlyEqual(Esum, Eavg.value * Eavg.weightSum, 1.0e-10), "open clover average over the V3*(Lt-2) interior sites");

    if (Lt >= 7) {
      const std::vector<Real> q5 = Loops::TimesliceTopologicalCharge5Li(U);
      const std::vector<Real> o5 = Loops::TimesliceTopologicalCharge5Li(U, open);
      for (int x0 = 0; x0 < Lt; ++x0) {
        const bool in3 = (x0 >= 3) && (x0 <= Lt - 4);
        check(in3 ? nearlyEqual(o5[x0], q5[x0], 1.0e-10) : (o5[x0] == 0.0), "open 5Li slice: periodic value inside 3<=x0<=T-3, zero outside");
      }
      check(nearlyEqual(Loops::TopologicalCharge5Li(U, periodic), Loops::TopologicalCharge5Li(U), 1.0e-10), "periodic 5Li Q = legacy");
    }

    // No wrap-around: the unused forward temporal links at x0=T do not enter.
    {
      LatticeGaugeField W(&grid), Uw(&grid);
      SU<Nc>::HotConfiguration(pRNG, W);
      LatticeColourMatrix Ut = PeekIndex<LorentzIndex>(U, t);
      LatticeColourMatrix Wt = PeekIndex<LorentzIndex>(W, t);
      LatticeInteger x0(&grid);
      LatticeCoordinate(x0, t);
      Ut = where(x0 == Integer(Lt - 1), Wt, Ut);
      Uw = U;
      PokeIndex<LorentzIndex>(Uw, Ut, t);

      const std::vector<Real> w1 = Loops::TimesliceTopologicalCharge(Uw, open);
      const std::vector<Real> w12 = Loops::TimesliceTopologicalChargeMxN(Uw, 1, 2, open);
      const GaugeLoopSliceAverage Ew = Loops::cloverEnergyDensitySliceAverage(Uw, open);
      for (int s = 0; s < Lt; ++s) {
        check(nearlyEqual(w1[s], o1[s], 1.0e-12) && nearlyEqual(w12[s], o12[s], 1.0e-12) && nearlyEqual(Ew.value[s], Eo.value[s], 1.0e-12),
              "open field-strength observables ignore the x0=T temporal link");
      }
      for (const auto &w : weights) {
        const GaugeLoopSliceAverage a = Loops::plaquetteSliceAverage(U, open, w);
        const GaugeLoopSliceAverage b = Loops::plaquetteSliceAverage(Uw, open, w);
        for (int s = 0; s < Lt; ++s) check(nearlyEqual(a.value[s], b.value[s], 1.0e-12), "open plaquette slices ignore the x0=T temporal link");
      }
    }

    // Unit gauge: no field strength.
    {
      LatticeGaugeField Uc(&grid);
      SU<Nc>::ColdConfiguration(pRNG, Uc);
      check(std::abs(Loops::TopologicalCharge(Uc, open)) < 1.0e-12, "unit gauge Q = 0");
      check(std::abs(Loops::cloverEnergyDensityAverage(Uc, open).value) < 1.0e-12, "unit gauge clover density = 0");
    }
  }

  std::cout << GridLogMessage << "all open-boundary loop checks passed" << std::endl;
  Grid_finalize();
  return 0;
}
