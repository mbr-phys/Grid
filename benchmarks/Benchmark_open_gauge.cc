/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_open_gauge.cc

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
////////////////////////////////////////////////////////////////////////////////
// Cost of open temporal gauge boundaries compared with periodic ones, on one
// lattice and one gauge field.  Every kernel is timed in up to three variants:
//
//   P  legacy periodic code (optimised staples, avgPlaquette, ...)
//   G  generic path code with periodic parameters (GaugeLoopPath /
//      OpenGaugeBoundary with open = false): cost of the implementation alone
//   O  generic path code with open boundaries: adds masks and weights
//
// G/P measures the algorithmic overhead of the generic path products, O/G the
// extra cost of the boundary handling itself.  Before timing, G is checked to
// reproduce P, so that only correct code is timed.
//
// Kernels: gauge force and action (Wilson, Symanzik) -- the force is what an
// open-boundary gradient flow evaluates in every Runge-Kutta stage -- and the
// measurements: plaquette and rectangle averages, plaquette time-slice profile,
// clover energy density, topological charge (1x1, 5Li), and the construction
// of the loop masks/weights.  No flow integrator is timed: Grid's WilsonFlow
// does not implement the openQCD open-boundary flow convention.
//
// Options: --iterations N (default 10)  --warmup N (default 2)  --cG x (default 1)
// plus the usual --grid, --mpi, --threads, --accelerator-threads.
////////////////////////////////////////////////////////////////////////////////
#include <Grid/Grid.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <numeric>

using namespace Grid;

typedef PeriodicGimplR            Gimpl;
typedef WilsonLoops<Gimpl>        Loops;
typedef GaugeLoopPath<Gimpl>      Path;
typedef OpenGaugeBoundary<Gimpl>  Boundary;

// Results are accumulated here so that no timed call can be optimised away.
static RealD sink = 0.0;

struct Timing {
  bool  valid = false;
  RealD mean = 0.0, min = 0.0, max = 0.0;  // ms per call
  RealD peakMB = 0.0;                      // extra peak allocation per call, MB per rank
};

struct Row {
  std::string name;
  Timing P, G, O;
};

static Timing measure(GridBase *grid, MemoryStats &stats, int warmup, int iterations,
                      const std::function<void(void)> &kernel) {
  Timing result;
  for (int i = 0; i < warmup; ++i) kernel();

  // Extra peak memory of one call.
  const size_t base = stats.currentlyAllocated;
  stats.maxAllocated = base;
  kernel();
  result.peakMB = RealD(stats.maxAllocated - base) / (1024.0 * 1024.0);

  std::vector<RealD> t(iterations);
  for (int i = 0; i < iterations; ++i) {
    grid->Barrier();
    RealD t0 = usecond();
    kernel();
    grid->Barrier();
    t[i] = (usecond() - t0) / 1000.0;
  }
  result.valid = true;
  result.mean = std::accumulate(t.begin(), t.end(), 0.0) / iterations;
  result.min = *std::min_element(t.begin(), t.end());
  result.max = *std::max_element(t.begin(), t.end());
  return result;
}

static std::string cell(const Timing &t) {
  std::stringstream s;
  if (t.valid) s << std::fixed << std::setprecision(2) << t.mean << " [" << t.min << "]";
  else s << "-";
  return s.str();
}

static std::string ratio(const Timing &a, const Timing &b) {
  std::stringstream s;
  if (a.valid && b.valid && b.mean > 0.0) s << std::fixed << std::setprecision(2) << a.mean / b.mean;
  else s << "-";
  return s.str();
}

static std::string memory(const Timing &t) {
  std::stringstream s;
  if (t.valid) s << std::fixed << std::setprecision(1) << t.peakMB;
  else s << "-";
  return s.str();
}

static void require(bool ok, const std::string &what) {
  std::cout << GridLogMessage << (ok ? "check passed: " : "CHECK FAILED: ") << what << std::endl;
  GRID_ASSERT(ok);
}

static bool nearlyEqual(RealD a, RealD b, RealD tol) {
  return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  int iterations = 10;
  int warmup = 2;
  RealD cG = 1.0;
  if (GridCmdOptionExists(argv, argv + argc, "--iterations"))
    iterations = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--iterations"));
  if (GridCmdOptionExists(argv, argv + argc, "--warmup"))
    warmup = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--warmup"));
  if (GridCmdOptionExists(argv, argv + argc, "--cG"))
    cG = std::stod(GridCmdOptionPayload(argv, argv + argc, "--cG"));
  GRID_ASSERT(iterations > 0 && warmup >= 0);

  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  const int t = Nd - 1;
  const int Lt = latt_size[t];
  GRID_ASSERT(Lt >= 3);

  std::cout << GridLogMessage << "Benchmark_open_gauge: lattice " << latt_size << " mpi " << mpi_layout
            << " simd " << simd_layout << " threads " << GridThread::GetThreads() << std::endl;
  std::cout << GridLogMessage << "iterations " << iterations << " warmup " << warmup << " cG " << cG << std::endl;

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({45, 12, 81, 9});
  LatticeGaugeField U(&grid), F(&grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  MemoryStats stats;
  MemoryProfiler::stats = &stats;

  const OpenGaugeBoundaryParameters periodic;
  const OpenGaugeBoundaryParameters open = OpenGaugeBoundaryParameters::Open(t, cG);

  // Actions.  Symanzik: c_plaq = beta(1 - 8 c1), c_rect = beta c1, c1 = -1/12.
  const RealD beta = 6.0;
  const RealD c1 = -1.0 / 12.0;
  const RealD c_plaq = beta * (1.0 - 8.0 * c1);
  const RealD c_rect = beta * c1;
  WilsonGaugeAction<Gimpl>   wilsonP(beta),   wilsonO(beta, open);
  SymanzikGaugeAction<Gimpl> symanzikP(beta), symanzikO(beta, open);
  Boundary genericPeriodic(periodic);
  const auto plaqLoops = Path::plaquetteLoops(beta);
  const auto symLoops = Path::plaquetteRectangleLoops(c_plaq, c_rect);
  const auto unitPlaq = Path::plaquetteLoops(1.0);
  const auto unitRect = Path::rectangleLoops(1.0);
  const auto allLoops = Path::plaquetteRectangleLoops(1.0, 1.0);

  //////////////////////////////////////////////////////////////////////////////
  // Preconditions: the generic periodic path (G) reproduces the legacy code (P).
  //////////////////////////////////////////////////////////////////////////////
  {
    LatticeGaugeField FP(&grid), FG(&grid), Fd(&grid);
    require(nearlyEqual(genericPeriodic.action(U, plaqLoops), wilsonP.S(U), 1.0e-10), "Wilson action G = P");
    require(nearlyEqual(genericPeriodic.action(U, symLoops), symanzikP.S(U), 1.0e-10), "Symanzik action G = P");
    wilsonP.deriv(U, FP);
    genericPeriodic.derivative(U, plaqLoops, FG);
    Fd = FG - FP;
    require(norm2(Fd) <= 1.0e-12 * (1.0 + norm2(FP)), "Wilson force G = P");
    symanzikP.deriv(U, FP);
    genericPeriodic.derivative(U, symLoops, FG);
    Fd = FG - FP;
    require(norm2(Fd) <= 1.0e-11 * (1.0 + norm2(FP)), "Symanzik force G = P");
    require(nearlyEqual(Path::loopAverage(U, unitPlaq, periodic, GaugeLoopWeights::geometric()).value, Loops::avgPlaquette(U), 1.0e-10), "plaquette G = P");
    require(nearlyEqual(Path::loopAverage(U, unitRect, periodic, GaugeLoopWeights::geometric()).value, Loops::avgRectangle(U), 1.0e-10), "rectangle G = P");
    require(nearlyEqual(Loops::TopologicalCharge(U, periodic), Loops::TopologicalCharge(U), 1.0e-10), "Q G = P");
  }

  std::vector<Row> rows;
  auto run = [&](const std::string &name,
                 const std::function<void(void)> &P,
                 const std::function<void(void)> &G,
                 const std::function<void(void)> &O) {
    Row r;
    r.name = name;
    std::cout << GridLogMessage << "timing " << name << std::endl;
    if (P) r.P = measure(&grid, stats, warmup, iterations, P);
    if (G) r.G = measure(&grid, stats, warmup, iterations, G);
    if (O) r.O = measure(&grid, stats, warmup, iterations, O);
    rows.push_back(r);
  };
  const std::function<void(void)> none;

  run("force Wilson",
      [&]() { wilsonP.deriv(U, F); sink += norm2(F); },
      [&]() { genericPeriodic.derivative(U, plaqLoops, F); sink += norm2(F); },
      [&]() { wilsonO.deriv(U, F); sink += norm2(F); });
  run("force Symanzik",
      [&]() { symanzikP.deriv(U, F); sink += norm2(F); },
      [&]() { genericPeriodic.derivative(U, symLoops, F); sink += norm2(F); },
      [&]() { symanzikO.deriv(U, F); sink += norm2(F); });
  run("action Wilson",
      [&]() { sink += wilsonP.S(U); },
      [&]() { sink += genericPeriodic.action(U, plaqLoops); },
      [&]() { sink += wilsonO.S(U); });
  run("action Symanzik",
      [&]() { sink += symanzikP.S(U); },
      [&]() { sink += genericPeriodic.action(U, symLoops); },
      [&]() { sink += symanzikO.S(U); });
  run("plaquette average",
      [&]() { sink += Loops::avgPlaquette(U); },
      [&]() { sink += Path::loopAverage(U, unitPlaq, periodic, GaugeLoopWeights::geometric()).value; },
      [&]() { sink += Loops::actionCompatiblePlaquetteAverage(U, open).value; });
  run("rectangle average",
      [&]() { sink += Loops::avgRectangle(U); },
      [&]() { sink += Path::loopAverage(U, unitRect, periodic, GaugeLoopWeights::geometric()).value; },
      [&]() { sink += Loops::actionCompatibleRectangleAverage(U, open).value; });
  run("plaquette slices (Split)",
      none,
      [&]() { sink += Loops::plaquetteSliceAverage(U, periodic, GaugeLoopWeights::treeLevel()).value[0]; },
      [&]() { sink += Loops::plaquetteSliceAverage(U, open, GaugeLoopWeights::treeLevel()).value[0]; });
  run("clover energy density",
      [&]() { sink += WilsonFlow<Gimpl>::energyDensityCloverleaf(1.0, U); },
      [&]() { sink += Loops::cloverEnergyDensityAverage(U, periodic).value; },
      [&]() { sink += Loops::cloverEnergyDensityAverage(U, open).value; });
  run("topological charge 1x1",
      [&]() { sink += Loops::TopologicalCharge(U); },
      [&]() { sink += Loops::TopologicalCharge(U, periodic); },
      [&]() { sink += Loops::TopologicalCharge(U, open); });
  if (Lt >= 7) {
    run("topological charge 5Li slices",
        [&]() { sink += Loops::TimesliceTopologicalCharge5Li(U)[0]; },
        [&]() { sink += Loops::TimesliceTopologicalCharge5Li(U, periodic)[0]; },
        [&]() { sink += Loops::TimesliceTopologicalCharge5Li(U, open)[0]; });
  }
  run("loop weights (plaq+rect)",
      none,
      [&]() { for (const auto &loop : allLoops) sink += real(TensorRemove(sum(Path::loopWeight(&grid, loop.path, periodic, GaugeLoopWeights::action(periodic))))); },
      [&]() { for (const auto &loop : allLoops) sink += real(TensorRemove(sum(Path::loopWeight(&grid, loop.path, open, GaugeLoopWeights::action(open))))); });

  MemoryProfiler::stats = nullptr;

  //////////////////////////////////////////////////////////////////////////////
  // Report
  //////////////////////////////////////////////////////////////////////////////
  const RealD sites = grid.gSites();
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "time per call in ms: mean [min] over " << iterations << " calls; ratios of means" << std::endl;
  std::cout << GridLogMessage << std::left << std::setw(32) << "kernel"
            << std::setw(20) << "P" << std::setw(20) << "G" << std::setw(20) << "O"
            << std::setw(8) << "G/P" << std::setw(8) << "O/P" << std::setw(8) << "O/G"
            << std::setw(12) << "O ns/site" << std::endl;
  for (const auto &r : rows) {
    std::stringstream perSite;
    if (r.O.valid) perSite << std::fixed << std::setprecision(2) << r.O.mean * 1.0e6 / sites;
    std::cout << GridLogMessage << std::left << std::setw(32) << r.name
              << std::setw(20) << cell(r.P) << std::setw(20) << cell(r.G) << std::setw(20) << cell(r.O)
              << std::setw(8) << ratio(r.G, r.P) << std::setw(8) << ratio(r.O, r.P) << std::setw(8) << ratio(r.O, r.G)
              << std::setw(12) << perSite.str() << std::endl;
  }
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "extra peak lattice allocation per call, MB per rank" << std::endl;
  std::cout << GridLogMessage << std::left << std::setw(32) << "kernel"
            << std::setw(12) << "P" << std::setw(12) << "G" << std::setw(12) << "O" << std::endl;
  for (const auto &r : rows) {
    std::cout << GridLogMessage << std::left << std::setw(32) << r.name
              << std::setw(12) << memory(r.P) << std::setw(12) << memory(r.G) << std::setw(12) << memory(r.O)
              << std::endl;
  }
  std::cout << GridLogMessage << "(checksum " << sink << ")" << std::endl;

  Grid_finalize();
  return 0;
}
