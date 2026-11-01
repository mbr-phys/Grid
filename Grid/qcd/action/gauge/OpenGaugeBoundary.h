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
// The generic loop-path force handles any loop family; the plaquette force also has a fast path on the padded
// staple stencil of the periodic actions (plaquetteDerivative).
template <class Gimpl> class OpenGaugeBoundary {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

private:
  // Loop lists come from GaugeLoopPath<Gimpl> directly, e.g. GaugeLoopPath<Gimpl>::plaquetteLoops(beta).
  typedef GaugeLoopPath<Gimpl> Path;
  typedef typename Path::Loop Loop;

  // Padded-cell workspace holding the plaquette staple stencil only
  class StapleWorkspace : public WilsonLoops<Gimpl>::WilsonLoopPaddedWorkspace {
  public:
    StapleWorkspace(void) {
      this->addStencil(new typename WilsonLoops<Gimpl>::StaplePaddedAllWorkspace);
    }
  };

  // Field-independent data, built on first use and reused while the grid is unchanged:
  // weight fields keyed by their time-slice profile (loop weights depend on x0 only), and the staple stencil.
  // Shared by copies of the boundary, which have the same parameters.
  struct Cache {
    GridBase *grid = nullptr;
    std::map<std::vector<RealD>, std::unique_ptr<ComplexField>> fields;
    std::unique_ptr<StapleWorkspace> staples;
  };

public:
  explicit OpenGaugeBoundary(const OpenGaugeBoundaryParameters &parameters = OpenGaugeBoundaryParameters())
    : parameters_(parameters), cache_(std::make_shared<Cache>()) {
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

  // Loops with a zero coefficient (e.g. the rectangles of a plaquette-only action) are skipped
  RealD action(const GaugeField &U, const std::vector<Loop> &loops) const {
    GridBase *grid = U.Grid();
    std::vector<GaugeLinkField> links = Path::extractLinks(U);
    const GaugeLoopWeights w = GaugeLoopWeights::action(parameters_);
    const int Lt = grid->GlobalDimensions()[parameters_.timeDir];
    GaugeLinkField transporter(grid);
    ComplexField reTrace(grid), density(grid);
    RealD result = 0.0;
    for (const auto &loop : loops) {
      if (loop.coefficient == 0.0) continue;
      Path::pathProduct(transporter, links, loop.path, Coordinate(Nd, 0));
      reTrace = real(trace(transporter));
      density = RealD(Gimpl::num_colours) - reTrace;
      const ComplexField *weight = profileField(grid, Path::loopWeightProfile(Lt, loop.path, parameters_, w));
      if (weight) density = (*weight) * density;
      result += loop.coefficient * real(TensorRemove(sum(density))) / RealD(Gimpl::num_colours);
    }
    return result;
  }

  // Loops with a zero coefficient are skipped
  void derivative(const GaugeField &U, const std::vector<Loop> &loops, GaugeField &dSdU) const {
    GridBase *grid = U.Grid();
    std::vector<GaugeLinkField> links = Path::extractLinks(U);
    std::vector<GaugeLinkField> staples(Nd, grid);
    GaugeLinkField staple(grid);
    for (int mu = 0; mu < Nd; ++mu) staples[mu] = Zero();

    for (const auto &loop : loops) {
      if (loop.coefficient == 0.0) continue;
      accumulateLoopStaples(links, loop, staples, staple);
    }

    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField force = Ta(links[mu] * staples[mu]) * (0.5 / RealD(Gimpl::num_colours));
      PokeIndex<LorentzIndex>(dSdU, force, mu);
    }
  }

  // Force of the plaquette action, equal to derivative(U, plaquetteLoops(c_plaq), dSdU), on the padded staple
  // stencil of the periodic actions. Every plaquette the open boundary removes contains an unused link
  // U_t(x0=T), and every plaquette containing one is removed, so zeroing those links removes exactly the
  // dropped plaquettes from the periodic staples. Only the spatial staples on x0=0 and x0=T then need the
  // extra weight cG/2 - 1. The staples of an unused link all contain another unused link, so its force is 0.
  void plaquetteDerivative(const GaugeField &U, RealD c_plaq, GaugeField &dSdU) const {
    GridBase *grid = U.Grid();
    GridCartesian *cartesian = dynamic_cast<GridCartesian *>(grid);
    if (!isOpen() || !cartesian) {
      derivative(U, Path::plaquetteLoops(c_plaq), dSdU);
      return;
    }

    const int t = parameters_.timeDir;
    const int Lt = grid->GlobalDimensions()[t];
    GRID_ASSERT(Lt >= 2);
    std::vector<GaugeLinkField> links = Path::extractLinks(U);

    // links with U_t(x0=T) set to zero
    std::vector<GaugeLinkField> cut(links);
    std::vector<RealD> keep(Lt, 1.0);
    keep[Lt - 1] = 0.0;
    cut[t] = (*profileField(grid, keep)) * links[t];

    Cache &c = cache(grid);
    if (!c.staples) c.staples.reset(new StapleWorkspace);
    const PaddedCell &cell = c.staples->getPaddedCell(cartesian);
    CshiftImplGauge<Gimpl> cshiftImpl;
    std::vector<GaugeLinkField> padded(Nd, cell.grids.back());
    for (int mu = 0; mu < Nd; ++mu) padded[mu] = cell.Exchange(cut[mu], cshiftImpl);
    std::vector<GaugeLinkField> staples(Nd, grid);
    WilsonLoops<Gimpl>::StaplePaddedAll(staples, padded, cell, c.staples->getStencil(0, cartesian));

    const RealD boundary = GaugeLoopWeights::action(parameters_).boundary;
    if (boundary != 1.0) {
      std::vector<RealD> edge(Lt, 0.0);
      edge[0] = boundary - 1.0;
      edge[Lt - 1] = boundary - 1.0;
      const ComplexField &correction = *profileField(grid, edge);
      GaugeLinkField spatial(grid);

      for (int mu = 0; mu < Nd; ++mu) {
        if (mu == t) continue;
        spatial = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if ((nu == mu) || (nu == t)) continue;
          // the two staples of WilsonLoops::Staple in the (mu, nu) plane
          spatial += Gimpl::ShiftStaple(
                       Gimpl::CovShiftForward(links[nu], nu,
                         Gimpl::CovShiftBackward(links[mu], mu, Gimpl::CovShiftIdentityBackward(links[nu], nu))),
                       mu);
          spatial += Gimpl::ShiftStaple(
                       Gimpl::CovShiftBackward(links[nu], nu, Gimpl::CovShiftBackward(links[mu], mu, links[nu])),
                       mu);
        }
        staples[mu] = staples[mu] + correction * spatial;
      }
    }

    const RealD factor = 0.5 * c_plaq / RealD(Gimpl::num_colours);
    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField force = Ta(links[mu] * staples[mu]) * factor;
      PokeIndex<LorentzIndex>(dSdU, force, mu);
    }
  }

private:
  OpenGaugeBoundaryParameters parameters_;
  std::shared_ptr<Cache> cache_;

  Cache &cache(GridBase *grid) const {
    if (cache_->grid != grid) {
      cache_->grid = grid;
      cache_->fields.clear();
    }
    return *cache_;
  }

  // Cached field of a time-slice profile; nullptr if the profile is 1 everywhere (nothing to multiply)
  const ComplexField *profileField(GridBase *grid, const std::vector<RealD> &profile) const {
    bool ones = true;
    for (auto v : profile) ones = ones && (v == 1.0);
    if (ones) return nullptr;

    Cache &c = cache(grid);
    auto it = c.fields.find(profile);
    if (it == c.fields.end()) {
      std::unique_ptr<ComplexField> field(new ComplexField(Path::profileField(grid, parameters_.timeDir, profile)));
      it = c.fields.emplace(profile, std::move(field)).first;
    }
    return it->second.get();
  }

  void accumulateLoopStaples(const std::vector<GaugeLinkField> &links, const Loop &loop,
                             std::vector<GaugeLinkField> &staples, GaugeLinkField &staple) const {
    GridBase *grid = links[0].Grid();
    const std::vector<int> &path = loop.path;
    const int n = path.size();
    const int t = parameters_.timeDir;
    const int Lt = grid->GlobalDimensions()[t];
    const std::vector<RealD> base = Path::loopWeightProfile(Lt, path, parameters_, GaugeLoopWeights::action(parameters_));
    std::vector<RealD> shifted(Lt);
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
      Path::pathProduct(staple, links, rest, initial);

      // The loop through this link at x starts at x - target: its weight is the base profile at x0 - target[t]
      for (int x0 = 0; x0 < Lt; ++x0) shifted[x0] = base[((x0 - target[t]) % Lt + Lt) % Lt];
      const ComplexField *weight = profileField(grid, shifted);
      if (weight) staples[mu] = staples[mu] + loop.coefficient * (*weight) * staple;
      else        staples[mu] = staples[mu] + loop.coefficient * staple;

      before[mu] += (step > 0) ? 1 : -1;
    }

    for (int d = 0; d < Nd; ++d) GRID_ASSERT(before[d] == 0);
  }
};

NAMESPACE_END(Grid);

#endif
