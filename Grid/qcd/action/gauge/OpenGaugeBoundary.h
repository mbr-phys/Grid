/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./Grid/qcd/action/gauge/OpenGaugeBoundary.h

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
#ifndef GRID_QCD_OPEN_GAUGE_BOUNDARY_H
#define GRID_QCD_OPEN_GAUGE_BOUNDARY_H

#include <Grid/qcd/utils/GaugeLoopPath.h>

NAMESPACE_BEGIN(Grid);

// Gauge action and force for open temporal boundary conditions (a la openQCD), built on the loop masks and weights of GaugeLoopPath
// Spatial loops on the boundary slices carry the action weight cG/2
// A small correctness-first implementation; the periodic actions keep their optimised paths.
template <class Gimpl> class OpenGaugeBoundary {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

private:
  // Loop lists come from GaugeLoopPath<Gimpl> directly, e.g. GaugeLoopPath<Gimpl>::plaquetteLoops(beta).
  typedef GaugeLoopPath<Gimpl> Path;
  typedef typename Path::Loop Loop;

public:
  explicit OpenGaugeBoundary(const OpenGaugeBoundaryParameters &parameters = OpenGaugeBoundaryParameters())
    : parameters_(parameters) {
    Path::checkParameters(parameters_);
  }

  const OpenGaugeBoundaryParameters &Parameters(void) const { return parameters_; }

  bool isOpen(void) const { return parameters_.open; }

  std::string LogParameters(void) const {
    // Preserve the existing action log output for the default periodic case.
    if (!isOpen()) return std::string();

    std::stringstream s;
    s << GridLogMessage << "[GaugeBoundary] type: open" << std::endl;
    s << GridLogMessage << "[GaugeBoundary] time direction: " << parameters_.timeDir << std::endl;
    s << GridLogMessage << "[GaugeBoundary] cG: " << parameters_.cG << std::endl;
    return s.str();
  }

  // Action weight of a loop: geometric mask, and cG/2 for purely spatial loops on x0=0 and x0=T
  ComplexField loopWeight(GridBase *grid, const std::vector<int> &path) const {
    return Path::loopWeight(grid, path, parameters_, GaugeLoopWeights::action(parameters_));
  }

  RealD action(const GaugeField &U, const std::vector<Loop> &loops) const {
    GridBase *grid = U.Grid();
    std::vector<GaugeLinkField> links = Path::extractLinks(U);
    const GaugeLoopWeights w = GaugeLoopWeights::action(parameters_);
    ComplexField weight(grid), reTrace(grid), density(grid);
    RealD result = 0.0;
    for (const auto &loop : loops) {
      Path::loopWeightAndTrace(weight, reTrace, links, loop.path, parameters_, w);
      density = weight * (RealD(Gimpl::num_colours) - reTrace);
      result += loop.coefficient * real(TensorRemove(sum(density))) / RealD(Gimpl::num_colours);
    }
    return result;
  }

  void derivative(const GaugeField &U, const std::vector<Loop> &loops, GaugeField &dSdU) const {
    GridBase *grid = U.Grid();
    std::vector<GaugeLinkField> links = Path::extractLinks(U);
    std::vector<GaugeLinkField> staples(Nd, grid);
    for (int mu = 0; mu < Nd; ++mu) staples[mu] = Zero();

    for (const auto &loop : loops) {
      ComplexField baseWeight = loopWeight(grid, loop.path);
      accumulateLoopStaples(links, loop.path, baseWeight, loop.coefficient, staples);
    }

    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField force = Ta(links[mu] * staples[mu]) * (0.5 / RealD(Gimpl::num_colours));
      PokeIndex<LorentzIndex>(dSdU, force, mu);
    }
  }

private:
  OpenGaugeBoundaryParameters parameters_;

  static void accumulateLoopStaples(const std::vector<GaugeLinkField> &links, const std::vector<int> &path,
                                    const ComplexField &baseWeight, RealD coefficient,
                                    std::vector<GaugeLinkField> &staples) {
    const int n = path.size();
    Coordinate before(Nd, 0);

    for (int i = 0; i < n; ++i) {
      const int step = path[i];
      const int mu = std::abs(step) - 1;
      Coordinate target = before;
      std::vector<int> rest;
      rest.reserve(n - 1);

      if (step > 0) {
        for (int j = 1; j < n; ++j) rest.push_back(path[(i + j) % n]);
      } else {
        target[mu]--;
        // Reverse the closed path, starting with the inverse of this backward step.
        // The remaining reversed steps form the staple.
        for (int j = 1; j < n; ++j) {
          int index = (i - j + n) % n;
          rest.push_back(-path[index]);
        }
      }

      Coordinate initial(Nd, 0);
      initial[mu] = 1;
      GaugeLinkField staple(baseWeight.Grid());
      Path::pathProduct(staple, links, rest, initial);

      Coordinate weightShift(Nd, 0);
      for (int d = 0; d < Nd; ++d) weightShift[d] = -target[d];

      ComplexField linkWeight = Path::shiftScalar(baseWeight, weightShift);
      staples[mu] = staples[mu] + coefficient * linkWeight * staple;

      before[mu] += (step > 0) ? 1 : -1;
    }

    for (int d = 0; d < Nd; ++d) GRID_ASSERT(before[d] == 0);
  }
};

NAMESPACE_END(Grid);

#endif
