/*************************************************************************************

Tests for CLS/openQCD temporal gauge boundary conditions.

*************************************************************************************/
#include <Grid/Grid.h>

using namespace Grid;

template<class Action>
RealD forceCheck(Action &action, LatticeGaugeField &U, GridParallelRNG &pRNG,
                 RealD dt)
{
  GridBase *grid = U.Grid();
  LatticeGaugeField force(grid), momentum(grid), Uprime(grid);
  LatticeColourMatrix p_mu(grid);

  action.deriv(U, force);
  for (int mu = 0; mu < Nd; ++mu) {
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, p_mu);
    PokeIndex<LorentzIndex>(momentum, p_mu, mu);
  }

  Uprime = U;
  autoView(Uprime_v, Uprime, CpuWrite);
  autoView(U_v, U, CpuRead);
  autoView(momentum_v, momentum, CpuRead);
  thread_foreach(ss, U_v, {
    for (int mu = 0; mu < Nd; ++mu)
      Uprime_v[ss](mu) = U_v[ss](mu) +
                         momentum_v[ss](mu) * U_v[ss](mu) * dt;
  });

  RealD actual = action.S(Uprime) - action.S(U);
  LatticeComplex predicted_density(grid);
  predicted_density = Zero();
  for (int mu = 0; mu < Nd; ++mu) {
    auto p = PeekIndex<LorentzIndex>(momentum, mu);
    auto f = PeekIndex<LorentzIndex>(force, mu);
    predicted_density = predicted_density - 2.0 * dt * trace(p * f);
  }
  RealD predicted = real(TensorRemove(sum(predicted_density)));
  RealD error = std::abs(actual - predicted);
  std::cout << GridLogMessage << action.action_name()
            << " actual dS " << actual
            << " predicted dS " << predicted
            << " error " << error << std::endl;
  return error;
}

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({45, 12, 81, 9});

  LatticeGaugeField U(&grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  // The generic path implementation must reproduce the established periodic
  // Wilson action and force before it is trusted for the open case.
  const RealD beta = 1.7;
  WilsonGaugeActionR periodic(beta);
  OpenGaugeBoundary<PeriodicGimplR> periodic_paths;
  RealD S_periodic = periodic.S(U);
  RealD S_paths = periodic_paths.action(
      U, OpenGaugeBoundary<PeriodicGimplR>::plaquetteLoops(beta));
  std::cout << GridLogMessage << "periodic Wilson action difference "
            << S_paths - S_periodic << std::endl;
  GRID_ASSERT(std::abs(S_paths - S_periodic) < 1.0e-9 * (1.0 + std::abs(S_periodic)));

  LatticeGaugeField F_periodic(&grid), F_paths(&grid), F_diff(&grid);
  periodic.deriv(U, F_periodic);
  periodic_paths.derivative(
      U, OpenGaugeBoundary<PeriodicGimplR>::plaquetteLoops(beta), F_paths);
  F_diff = F_paths - F_periodic;
  std::cout << GridLogMessage << "periodic Wilson force difference norm2 "
            << norm2(F_diff) << std::endl;
  GRID_ASSERT(norm2(F_diff) < 1.0e-12 * (1.0 + norm2(F_periodic)));

  const RealD c_plaq = 2.3;
  const RealD c_rect = -0.17;
  PlaqPlusRectangleAction<PeriodicGimplR> periodic_rect(c_plaq, c_rect);
  RealD S_rect_periodic = periodic_rect.S(U);
  RealD S_rect_paths = periodic_paths.action(
      U, OpenGaugeBoundary<PeriodicGimplR>::plaquetteRectangleLoops(
             c_plaq, c_rect));
  std::cout << GridLogMessage << "periodic plaquette+rectangle action difference "
            << S_rect_paths - S_rect_periodic << std::endl;
  GRID_ASSERT(std::abs(S_rect_paths - S_rect_periodic) <
              1.0e-9 * (1.0 + std::abs(S_rect_periodic)));

  periodic_rect.deriv(U, F_periodic);
  periodic_paths.derivative(
      U, OpenGaugeBoundary<PeriodicGimplR>::plaquetteRectangleLoops(
             c_plaq, c_rect),
      F_paths);
  F_diff = F_paths - F_periodic;
  std::cout << GridLogMessage
            << "periodic plaquette+rectangle force difference norm2 "
            << norm2(F_diff) << std::endl;
  GRID_ASSERT(norm2(F_diff) < 1.0e-11 * (1.0 + norm2(F_periodic)));

  GRID_ASSERT(latt_size[Nd - 1] >= 3);
  const RealD test_cG = 1.4;
  OpenGaugeBoundaryParameters weight_parameters =
      OpenGaugeBoundaryParameters::Open(Nd - 1, test_cG);
  OpenGaugeBoundary<PeriodicGimplR> weighted_boundary(weight_parameters);
  auto spatial_weight = weighted_boundary.loopWeight(
      &grid, {1, 2, -1, -2});
  auto temporal_weight = weighted_boundary.loopWeight(
      &grid, {1, Nd, -1, -Nd});
  const RealD spatial_volume = grid.gSites() / latt_size[Nd - 1];
  const RealD expected_spatial_weight =
      spatial_volume * (latt_size[Nd - 1] - 2 + test_cG);
  const RealD expected_temporal_weight =
      spatial_volume * (latt_size[Nd - 1] - 1);
  GRID_ASSERT(std::abs(real(TensorRemove(sum(spatial_weight))) -
                       expected_spatial_weight) < 1.0e-10);
  GRID_ASSERT(std::abs(real(TensorRemove(sum(temporal_weight))) -
                       expected_temporal_weight) < 1.0e-10);

  OpenGaugeBoundaryParameters obc =
      OpenGaugeBoundaryParameters::Open(Nd - 1, 1.0);
  WilsonGaugeAction<PeriodicGimplR> open_wilson(beta, obc);
  IwasakiGaugeAction<PeriodicGimplR> open_iwasaki(beta, obc);

  GRID_ASSERT(forceCheck(open_wilson, U, pRNG, 1.0e-6) < 1.0e-7);
  GRID_ASSERT(forceCheck(open_iwasaki, U, pRNG, 1.0e-6) < 1.0e-7);

  // The unused forward temporal links at x0=T occur in no retained loop and
  // must therefore have exactly zero gauge force even before an HMC filter is
  // introduced.
  LatticeGaugeField F_open(&grid);
  open_iwasaki.deriv(U, F_open);
  LatticeColourMatrix Ft = PeekIndex<LorentzIndex>(F_open, Nd - 1);
  LatticeColourMatrix zero(&grid); zero = Zero();
  LatticeInteger x0(&grid); LatticeCoordinate(x0, Nd - 1);
  Ft = where(x0 == Integer(latt_size[Nd - 1] - 1), Ft, zero);
  std::cout << GridLogMessage << "dummy temporal-link force norm2 "
            << norm2(Ft) << std::endl;
  GRID_ASSERT(norm2(Ft) == 0.0);

  Grid_finalize();
  return 0;
}
