/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Validate FGMRES with the non-Hermitian Schur DiagTwo convention used by
Hadrons by default.

*************************************************************************************/
#include <cassert>
#include <Grid/Grid.h>

using namespace Grid;

namespace {
template <typename Matrix, typename Field>
RealD fullResidual(Matrix &mat, const Field &source, const Field &solution)
{
  Field residual(source.Grid());

  mat.M(solution, residual);
  residual = residual - source;
  return std::sqrt(norm2(residual) / norm2(source));
}
}

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  Coordinate lattice = GridDefaultLatt();
  Coordinate simd    = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi     = GridDefaultMpi();
  GridCartesian           grid(lattice, simd, mpi);
  GridRedBlackCartesian rbGrid(&grid);

  GridParallelRNG rng(&grid);
  rng.SeedFixedIntegers({1, 2, 3, 4});

  using Action = CompactWilsonCloverFermionD;
  using Field  = Action::FermionField;

  LatticeGaugeFieldD gauge(&grid);
  Field source(&grid);
  Field zeroGuessSolution(&grid);
  Field sourceGuessSolution(&grid);

  SU<Nc>::HotConfiguration(rng, gauge);
  gaussian(rng, source);
  zeroGuessSolution   = Zero();
  sourceGuessSolution = Zero();

  const RealD mass          = 0.5;
  const RealD cswR          = 1.0;
  const RealD cswT          = 1.0;
  const RealD cF            = 0.0;
  const RealD tolerance     = 1.0e-10;
  const int   maxIterations = 2000;
  const int   restartLength = 25;

  Action action(gauge, grid, rbGrid, mass, cswR, cswT, cF);
  TrivialPrecon<Field> identity;

  {
    FlexibleGeneralisedMinimalResidual<Field> fgmres(
      tolerance, maxIterations, identity, restartLength);
    NonHermitianSchurRedBlackDiagTwoSolve<Field> schurSolver(fgmres);
    ZeroGuesser<Field> guesser;

    schurSolver(action, source, zeroGuessSolution, guesser);
  }

  {
    FlexibleGeneralisedMinimalResidual<Field> fgmres(
      tolerance, maxIterations, identity, restartLength);
    NonHermitianSchurRedBlackDiagTwoSolve<Field> schurSolver(fgmres);
    SourceGuesser<Field> guesser;

    schurSolver(action, source, sourceGuessSolution, guesser);
  }

  const RealD zeroGuessResidual =
    fullResidual(action, source, zeroGuessSolution);
  const RealD sourceGuessResidual =
    fullResidual(action, source, sourceGuessSolution);
  Field difference(&grid);
  difference = zeroGuessSolution - sourceGuessSolution;
  const RealD relativeDifference =
    std::sqrt(norm2(difference) / norm2(zeroGuessSolution));

  std::cout << GridLogMessage
            << "Schur FGMRES zero-guess full residual: "
            << zeroGuessResidual << std::endl;
  std::cout << GridLogMessage
            << "Schur FGMRES source-guess full residual: "
            << sourceGuessResidual << std::endl;
  std::cout << GridLogMessage
            << "Schur FGMRES relative solution difference: "
            << relativeDifference << std::endl;

  assert(zeroGuessResidual <= 10.0 * tolerance);
  assert(sourceGuessResidual <= 10.0 * tolerance);
  assert(relativeDifference <= 100.0 * tolerance);

  Grid_finalize();
  return EXIT_SUCCESS;
}
