/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./Grid/qcd/utils/GaugeLoopPath.h

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
#ifndef GRID_QCD_GAUGE_LOOP_PATH_H
#define GRID_QCD_GAUGE_LOOP_PATH_H

#include <map>
#include <tuple>

NAMESPACE_BEGIN(Grid);

////////////////////////////////////////////////////////////////////////////////
// Closed gauge-link paths with optional open temporal boundary conditions
//
// A path is a list of steps +(mu+1) (forward) or -(mu+1) (backward).  With open
// boundaries a translated loop is present only if all of its vertices lie in
// [0,T] along the open direction (T = L_t - 1); this is the loop mask, which
// depends on geometry only.  Purely spatial loops on the two boundary slices
// x0=0 and x0=T then carry a boundary weight chosen by GaugeLoopWeights:
//
//   geometric()          1
//   treeLevel()          1/2
//   action(p)            cG/2
//
// With periodic boundaries the mask and weight are 1 everywhere, so every
// convention reduces to the usual periodic sum
////////////////////////////////////////////////////////////////////////////////

class OpenGaugeBoundaryParameters : public Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(OpenGaugeBoundaryParameters,
                                  bool, open,
                                  int, timeDir,
                                  RealD, cG);

  OpenGaugeBoundaryParameters(bool open_ = false,
                              int timeDir_ = Nd - 1,
                              RealD cG_ = 1.0)
    : open(open_), timeDir(timeDir_), cG(cG_) {}

  static OpenGaugeBoundaryParameters Open(int timeDir_ = Nd - 1, RealD cG_ = 1.0) {
    return OpenGaugeBoundaryParameters(true, timeDir_, cG_);
  }
};

// Weight of purely spatial loops on the boundary slices x0=0 and x0=T
struct GaugeLoopWeights {
  RealD boundary;

  static GaugeLoopWeights geometric(void) { return {1.0}; }
  static GaugeLoopWeights treeLevel(void) { return {0.5}; }
  static GaugeLoopWeights action(const OpenGaugeBoundaryParameters &p) { return {0.5 * p.cG}; }
};

// Weighted average of Re tr(loop)/Nc, and the sum of the weights used to normalise it
struct GaugeLoopAverage {
  RealD value;
  RealD weightSum;
};

// The same per time slice x0 = 0..L_t-1 of the open (or, if periodic, the last) direction
// value[x0] is 0 where weightSum[x0] is 0
struct GaugeLoopSliceAverage {
  std::vector<RealD> value;
  std::vector<RealD> weightSum;
};

// How a loop is attributed to time slices:
//   Origin: the whole loop to the slice of its starting site
//   Split:  equally to every slice its vertices span
//           For plaquettes this is: a space-like plaquette counts at its own slice,
//           a time-like one 1/2 at x0 and 1/2 at x0+1 (a la openQCD)
enum class GaugeLoopSliceAssignment { Origin, Split };

template <class Gimpl> class GaugeLoopPath {
public:
  typedef typename Gimpl::LinkField    GaugeLinkField;
  typedef typename Gimpl::Field        GaugeField;
  typedef typename Gimpl::ComplexField ComplexField;

  struct Loop {
    std::vector<int> path;
    RealD coefficient;
    std::string name;

    Loop(const std::vector<int> &path_, RealD coefficient_, const std::string &name_ = "")
      : path(path_), coefficient(coefficient_), name(name_) {}
  };

  ///////////////////////////////////////////////////////////////////////////
  // Loop families
  ///////////////////////////////////////////////////////////////////////////
  static std::vector<Loop> plaquetteLoops(RealD coefficient) {
    std::vector<Loop> loops;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        loops.emplace_back(std::vector<int>{mu + 1, nu + 1, -(mu + 1), -(nu + 1)}, coefficient, "plaquette");
      }
    }
    return loops;
  }

  // Both 2x1 orientations in every plane.
  static std::vector<Loop> rectangleLoops(RealD coefficient) {
    std::vector<Loop> loops;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        loops.emplace_back(std::vector<int>{mu + 1, mu + 1, nu + 1, -(mu + 1), -(mu + 1), -(nu + 1)}, coefficient, "rectangle");
        loops.emplace_back(std::vector<int>{mu + 1, nu + 1, nu + 1, -(mu + 1), -(nu + 1), -(nu + 1)}, coefficient, "rectangle");
      }
    }
    return loops;
  }

  static std::vector<Loop> plaquetteRectangleLoops(RealD c_plaq, RealD c_rect) {
    std::vector<Loop> loops = plaquetteLoops(c_plaq);
    std::vector<Loop> rect  = rectangleLoops(c_rect);
    loops.insert(loops.end(), rect.begin(), rect.end());
    return loops;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Masks and weights
  ///////////////////////////////////////////////////////////////////////////
  static bool isSpatial(const std::vector<int> &path, const int timeDir) {
    for (auto step : path) if (std::abs(step) - 1 == timeDir) return false;
    return true;
  }

  // 1 where the loop starting at that site lies entirely in [0,T] along the open direction, 0 otherwise
  // Identically 1 for periodic boundaries
  static ComplexField loopMask(GridBase *grid, const std::vector<int> &path,
                               const OpenGaugeBoundaryParameters &p) {
    ComplexField mask(grid);
    mask = 1.0;
    if (!p.open) return mask;

    checkParameters(p);
    const int t = p.timeDir;
    const int T = grid->GlobalDimensions()[t] - 1;
    GRID_ASSERT(T >= 1);

    int minimum, maximum;
    temporalRange(path, t, minimum, maximum);

    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    ComplexField zero(grid); zero = 0.0;

    if (minimum < 0) mask = where(x0 < Integer(-minimum), zero, mask);
    if (maximum > 0) mask = where(x0 > Integer(T - maximum), zero, mask);
    return mask;
  }

  // Loop mask times the boundary weight for purely spatial loops on x0=0 and x0=T; bulk loops carry weight 1
  static ComplexField loopWeight(GridBase *grid, const std::vector<int> &path,
                                 const OpenGaugeBoundaryParameters &p,
                                 const GaugeLoopWeights &w) {
    ComplexField weight = loopMask(grid, path, p);
    if (!p.open || !isSpatial(path, p.timeDir)) return weight;

    const int t = p.timeDir;
    const int T = grid->GlobalDimensions()[t] - 1;
    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    ComplexField boundaryWeight(grid);
    boundaryWeight = w.boundary;
    weight = where((x0 == Integer(0)) || (x0 == Integer(T)), boundaryWeight, weight);
    return weight;
  }

  // loopWeight depends on the time slice x0 of the loop's starting site only: its value on each slice x0 = 0..Lt-1
  // (all ones if periodic)
  static std::vector<RealD> loopWeightProfile(const int Lt, const std::vector<int> &path,
                                              const OpenGaugeBoundaryParameters &p,
                                              const GaugeLoopWeights &w) {
    std::vector<RealD> profile(Lt, 1.0);
    if (!p.open) return profile;

    checkParameters(p);
    const int T = Lt - 1;
    GRID_ASSERT(T >= 1);

    int minimum, maximum;
    temporalRange(path, p.timeDir, minimum, maximum);
    const bool spatial = isSpatial(path, p.timeDir);
    for (int x0 = 0; x0 < Lt; ++x0) {
      if ((x0 < -minimum) || (x0 > T - maximum)) profile[x0] = 0.0;
      else if (spatial && ((x0 == 0) || (x0 == T))) profile[x0] = w.boundary;
    }
    return profile;
  }

  // Scalar field equal to profile[x0] on the time slice x0 of direction t
  // Built run by run (a run being consecutive slices with equal values), two where() per non-zero run
  static ComplexField profileField(GridBase *grid, const int t, const std::vector<RealD> &profile) {
    const int Lt = grid->GlobalDimensions()[t];
    GRID_ASSERT(int(profile.size()) == Lt);
    ComplexField field(grid), value(grid), lower(grid);
    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    field = Zero();
    for (int a = 0; a < Lt;) {
      int b = a;
      while ((b + 1 < Lt) && (profile[b + 1] == profile[a])) ++b;
      if (profile[a] != 0.0) {
        value = profile[a];
        lower = where(x0 > Integer(a - 1), value, field); // value on x0 >= a
        field = where(x0 < Integer(b + 1), lower, field); // and x0 <= b
      }
      a = b + 1;
    }
    return field;
  }

  static bool isOne(const std::vector<RealD> &profile) {
    for (auto v : profile) if (v != 1.0) return false;
    return true;
  }

  // Lowest and highest offset of the path's vertices along direction t, relative to its starting site (minimum <= 0 <= maximum)
  static void temporalRange(const std::vector<int> &path, const int t, int &minimum, int &maximum) {
    int displacement = 0;
    minimum = 0;
    maximum = 0;
    for (auto step : path) {
      int mu = std::abs(step) - 1;
      GRID_ASSERT(mu >= 0 && mu < Nd);
      if (mu == t) {
        displacement += (step > 0) ? 1 : -1;
        minimum = std::min(minimum, displacement);
        maximum = std::max(maximum, displacement);
      }
    }
    GRID_ASSERT(displacement == 0);
  }

  // Site mask for clover-based quantities (field strength, clover energy density, topological charge)
  // A clover at site x is built from leaves that reach up to `extent` links forwards and backwards in time
  // (extent 1 for the 1x1 clover, max(M,N) for MxN loops); it is complete only if all of them stay inside [0,T]
  // The mask is 1 on those sites and 0 elsewhere (1 everywhere if periodic); extent 1 gives 0 < x0 < T
  // Requires T >= 2*extent so that some site survives: Lt >= 3 for extent 1, Lt >= 5 for extent 2, Lt >= 7 for extent 3.
  static ComplexField interiorMask(GridBase *grid, const OpenGaugeBoundaryParameters &p, const int extent) {
    ComplexField mask(grid);
    mask = 1.0;
    if (!p.open || extent <= 0) return mask;

    checkParameters(p);
    const int t = p.timeDir;
    const int T = grid->GlobalDimensions()[t] - 1;
    GRID_ASSERT(T >= 2 * extent);

    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    ComplexField zero(grid); zero = 0.0;
    mask = where((x0 < Integer(extent)) || (x0 > Integer(T - extent)), zero, mask);
    return mask;
  }

  // Real part of the time-slice sums of a scalar field along the open direction (the last direction if periodic)
  static std::vector<RealD> sliceSums(const ComplexField &field, const OpenGaugeBoundaryParameters &p) {
    typedef typename ComplexField::scalar_object sobj;
    std::vector<sobj> sums;
    sliceSum(field, sums, p.timeDir);
    std::vector<RealD> out(sums.size());
    for (int x0 = 0; x0 < sums.size(); ++x0) out[x0] = TensorRemove(sums[x0]).real();
    return out;
  }

  // The weights are uniform in space, so their sums follow from the time-slice profiles alone
  static RealD loopWeightSum(GridBase *grid, const std::vector<Loop> &loops,
                             const OpenGaugeBoundaryParameters &p,
                             const GaugeLoopWeights &w) {
    const int Lt = grid->GlobalDimensions()[p.timeDir];
    const RealD V3 = RealD(grid->gSites()) / RealD(Lt);
    RealD weightSum = 0.0;
    for (const auto &loop : loops) {
      for (auto v : loopWeightProfile(Lt, loop.path, p, w)) weightSum += v * V3;
    }
    return weightSum;
  }

  // For the loop starting at each site x: its weight w(x) (loop mask times boundary weight) and the
  // unweighted real trace Re tr of its link product.  Shared by loopAverage, loopSliceAverage and the
  // open gauge action, so that action and observables use identical loops and weights.
  static void loopWeightAndTrace(ComplexField &weight, ComplexField &reTrace,
                                 const std::vector<GaugeLinkField> &links, const std::vector<int> &path,
                                 const OpenGaugeBoundaryParameters &p, const GaugeLoopWeights &w) {
    GridBase *grid = links[0].Grid();
    GaugeLinkField transporter(grid);
    weight = loopWeight(grid, path, p, w);
    pathProduct(transporter, links, path, Coordinate(Nd, 0));
    reTrace = real(trace(transporter));
  }

  ///////////////////////////////////////////////////////////////////////////
  // Weighted average of Re tr(loop)/Nc over a loop family
  // Loop coefficients are ignored: every loop of the family enters with
  // its boundary weight only
  ///////////////////////////////////////////////////////////////////////////
  // Loops with the same weight profile have their traces summed first, so each distinct weight field is built
  // once per call (plaquettes: 2, rectangles: 3); the weight sum follows from the profiles
  static GaugeLoopAverage loopAverage(const GaugeField &U, const std::vector<Loop> &loops,
                                      const OpenGaugeBoundaryParameters &p,
                                      const GaugeLoopWeights &w) {
    GridBase *grid = U.Grid();
    const int Lt = grid->GlobalDimensions()[p.timeDir];
    const RealD V3 = RealD(grid->gSites()) / RealD(Lt);
    std::vector<GaugeLinkField> links = extractLinks(U);
    GaugeLinkField transporter(grid);
    ComplexField reTrace(grid), density(grid);
    std::map<std::vector<RealD>, ComplexField> traces;
    RealD numerator = 0.0;
    RealD weightSum = 0.0;

    for (const auto &loop : loops) {
      const std::vector<RealD> profile = loopWeightProfile(Lt, loop.path, p, w);
      pathProduct(transporter, links, loop.path, Coordinate(Nd, 0));
      reTrace = real(trace(transporter));
      auto it = traces.find(profile);
      if (it == traces.end()) traces.emplace(profile, reTrace);
      else                    it->second = it->second + reTrace;
      for (auto v : profile) weightSum += v * V3;
    }
    for (const auto &c : traces) {
      if (isOne(c.first)) density = c.second;
      else                density = profileField(grid, p.timeDir, c.first) * c.second;
      numerator += real(TensorRemove(sum(density)));
    }
    GRID_ASSERT(weightSum > 0.0);

    GaugeLoopAverage result;
    result.value = numerator / (RealD(Gimpl::num_colours) * weightSum);
    result.weightSum = weightSum;
    return result;
  }

  ///////////////////////////////////////////////////////////////////////////
  // loopAverage resolved by time slice
  // The slice weight sums add up to the loopAverage weightSum, and
  // sum_x0 value*weightSum to its value*weightSum
  ///////////////////////////////////////////////////////////////////////////
  static GaugeLoopSliceAverage loopSliceAverage(const GaugeField &U, const std::vector<Loop> &loops,
                                                const OpenGaugeBoundaryParameters &p,
                                                const GaugeLoopWeights &w,
                                                const GaugeLoopSliceAssignment assignment = GaugeLoopSliceAssignment::Split) {
    checkParameters(p);
    GridBase *grid = U.Grid();
    const int t = p.timeDir;
    const int Lt = grid->GlobalDimensions()[t];
    const RealD V3 = RealD(grid->gSites()) / RealD(Lt);
    std::vector<GaugeLinkField> links = extractLinks(U);
    GaugeLinkField transporter(grid);
    ComplexField reTrace(grid), density(grid);
    // traces summed over the loops sharing a weight profile and slice offsets [minimum, maximum]
    std::map<std::tuple<std::vector<RealD>, int, int>, ComplexField> traces;
    std::vector<RealD> weightProfile(Lt, 0.0), numerator(Lt, 0.0);
    auto slice = [Lt](const int x0) { return ((x0 % Lt) + Lt) % Lt; };

    // Each contribution moves from its starting slice x0 to x0 + k for k in [minimum, maximum], with weight
    // 1/(maximum - minimum + 1); masked loops never reach past the open boundary, so the periodic shift does not
    // wrap anything that is present. Both are done on the slice sums, the weights' entirely from their profiles.
    for (const auto &loop : loops) {
      const std::vector<RealD> profile = loopWeightProfile(Lt, loop.path, p, w);
      int minimum = 0, maximum = 0;
      if (assignment == GaugeLoopSliceAssignment::Split) temporalRange(loop.path, t, minimum, maximum);
      const RealD share = 1.0 / RealD(maximum - minimum + 1);

      pathProduct(transporter, links, loop.path, Coordinate(Nd, 0));
      reTrace = real(trace(transporter));
      const auto key = std::make_tuple(profile, minimum, maximum);
      auto it = traces.find(key);
      if (it == traces.end()) traces.emplace(key, reTrace);
      else                    it->second = it->second + reTrace;

      for (int k = minimum; k <= maximum; ++k) {
        for (int x0 = 0; x0 < Lt; ++x0) weightProfile[x0] += share * V3 * profile[slice(x0 - k)];
      }
    }
    for (const auto &c : traces) {
      const std::vector<RealD> &profile = std::get<0>(c.first);
      const int minimum = std::get<1>(c.first), maximum = std::get<2>(c.first);
      const RealD share = 1.0 / RealD(maximum - minimum + 1);

      if (isOne(profile)) density = c.second;
      else                density = profileField(grid, t, profile) * c.second;
      const std::vector<RealD> sums = sliceSums(density, p);
      for (int k = minimum; k <= maximum; ++k) {
        for (int x0 = 0; x0 < Lt; ++x0) numerator[x0] += share * sums[slice(x0 - k)];
      }
    }

    GaugeLoopSliceAverage result;
    result.weightSum = weightProfile;
    result.value.resize(numerator.size());
    for (int x0 = 0; x0 < numerator.size(); ++x0) {
      const RealD W = result.weightSum[x0];
      result.value[x0] = (W > 0.0) ? numerator[x0] / (RealD(Gimpl::num_colours) * W) : 0.0;
    }
    return result;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Path products (periodic shifts; boundary handling is done by the mask)
  ///////////////////////////////////////////////////////////////////////////
  static std::vector<GaugeLinkField> extractLinks(const GaugeField &U) {
    std::vector<GaugeLinkField> links(Nd, U.Grid());
    for (int mu = 0; mu < Nd; ++mu) links[mu] = PeekIndex<LorentzIndex>(U, mu);
    return links;
  }

  static GaugeLinkField shiftGauge(const GaugeLinkField &in, const Coordinate &shift) {
    GaugeLinkField out = in;
    for (int mu = 0; mu < Nd; ++mu) {
      int n = shift[mu];
      int direction = (n >= 0) ? 1 : -1;
      for (int i = 0; i < std::abs(n); ++i) out = Gimpl::CshiftLink(out, mu, direction);
    }
    return out;
  }

  static ComplexField shiftScalar(const ComplexField &in, const Coordinate &shift) {
    ComplexField out = in;
    for (int mu = 0; mu < Nd; ++mu) {
      int n = shift[mu];
      int direction = (n >= 0) ? 1 : -1;
      for (int i = 0; i < std::abs(n); ++i) out = Cshift(out, mu, direction);
    }
    return out;
  }

  // Product of links along path, starting at x + displacement, as a field in x.
  // Two equivalent constructions, chosen per path by their number of single-site shifts (each a Cshift,
  // i.e. communication when the direction is split across ranks):
  //   per link:    each link shifted by its offset; cheapest for short closed loops (plaquette: 2 shifts)
  //   incremental: one shift of the running product per step; cheaper for longer open paths
  // On a tie the per-link construction is used.
  static void pathProduct(GaugeLinkField &out, const std::vector<GaugeLinkField> &links,
                          const std::vector<int> &path, const Coordinate &displacement) {
    if (path.empty()) {
      out = 1.0;
      return;
    }
    if (incrementalShifts(path, displacement) < perLinkShifts(path, displacement)) {
      pathProductIncremental(out, links, path, displacement);
    } else {
      pathProductPerLink(out, links, path, displacement);
    }
  }

  // Shifts needed when each link is shifted by its offset from x
  static int perLinkShifts(const std::vector<int> &path, Coordinate displacement) {
    int shifts = 0;
    for (auto step : path) {
      const int mu = std::abs(step) - 1;
      if (step < 0) displacement[mu]--;
      for (int d = 0; d < Nd; ++d) shifts += std::abs(displacement[d]);
      if (step > 0) displacement[mu]++;
    }
    return shifts;
  }

  // Shifts needed by the incremental construction: one per step, except a forward last step, plus the
  // shift of the result by the starting displacement
  static int incrementalShifts(const std::vector<int> &path, const Coordinate &displacement) {
    int shifts = int(path.size()) - ((path.back() > 0) ? 1 : 0);
    for (int d = 0; d < Nd; ++d) shifts += std::abs(displacement[d]);
    return shifts;
  }

  static void pathProductPerLink(GaugeLinkField &out, const std::vector<GaugeLinkField> &links,
                                 const std::vector<int> &path, Coordinate displacement) {
    bool first = true;
    for (auto step : path) {
      const int mu = std::abs(step) - 1;
      if (step > 0) {
        if (first) out = shiftGauge(links[mu], displacement);
        else       out = out * shiftGauge(links[mu], displacement);
        displacement[mu]++;
      } else {
        displacement[mu]--;
        if (first) out = shiftGauge(adj(links[mu]), displacement);
        else       out = out * shiftGauge(adj(links[mu]), displacement);
      }
      first = false;
    }
  }

  // Built from the end of the path: with R_k(y) the product of steps k.. starting at y,
  //   forward  step: R_k(y) = U_mu(y) R_{k+1}(y + mu)
  //   backward step: R_k(y) = U_mu(y - mu)^dag R_{k+1}(y - mu)
  static void pathProductIncremental(GaugeLinkField &out, const std::vector<GaugeLinkField> &links,
                                     const std::vector<int> &path, const Coordinate &displacement) {
    GaugeLinkField tmp(links[0].Grid());
    bool first = true;
    for (int k = path.size() - 1; k >= 0; --k) {
      const int step = path[k];
      const int mu = std::abs(step) - 1;
      if (step > 0) {
        if (first) {
          out = links[mu];
        } else {
          tmp = Gimpl::CshiftLink(out, mu, 1);
          out = links[mu] * tmp;
        }
      } else {
        if (first) tmp = adj(links[mu]);
        else       tmp = adj(links[mu]) * out;
        out = Gimpl::CshiftLink(tmp, mu, -1);
      }
      first = false;
    }
    bool shifted = false;
    for (int d = 0; d < Nd; ++d) shifted = shifted || (displacement[d] != 0);
    if (shifted) out = shiftGauge(out, displacement);
  }

  static void checkParameters(const OpenGaugeBoundaryParameters &p) {
    GRID_ASSERT(p.timeDir >= 0);
    GRID_ASSERT(p.timeDir < Nd);
    GRID_ASSERT(p.cG >= 0.0);
  }
};

NAMESPACE_END(Grid);

#endif
