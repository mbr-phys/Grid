/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Open temporal gauge boundary conditions in the CLS/openQCD convention.

*************************************************************************************/
#ifndef GRID_QCD_OPEN_GAUGE_BOUNDARY_H
#define GRID_QCD_OPEN_GAUGE_BOUNDARY_H

NAMESPACE_BEGIN(Grid);

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

  static OpenGaugeBoundaryParameters Open(int timeDir_ = Nd - 1,
                                              RealD cG_ = 1.0) {
    return OpenGaugeBoundaryParameters(true, timeDir_, cG_);
  }
};

// A small correctness-first implementation of the CLS/openQCD gauge boundary.
// Paths use +(mu+1) for a forward step and -(mu+1) for a backward step.
template<class Gimpl>
class OpenGaugeBoundary {
public:
  INHERIT_GIMPL_TYPES(Gimpl);


  struct Loop {
    std::vector<int> path;
    RealD coefficient;
    std::string name;

    Loop(const std::vector<int> &path_, RealD coefficient_,
         const std::string &name_ = "")
      : path(path_), coefficient(coefficient_), name(name_) {}
  };

  explicit OpenGaugeBoundary(
      const OpenGaugeBoundaryParameters &parameters =
          OpenGaugeBoundaryParameters())
      : parameters_(parameters) {
    GRID_ASSERT(parameters_.timeDir >= 0);
    GRID_ASSERT(parameters_.timeDir < Nd);
    GRID_ASSERT(parameters_.cG >= 0.0);
  }

  const OpenGaugeBoundaryParameters &Parameters(void) const {
    return parameters_;
  }

  bool isOpen(void) const { return parameters_.open; }

  std::string LogParameters(void) const {
    // Preserve the existing action log output for the default periodic case.
    if (!isOpen()) return std::string();

    std::stringstream s;
    s << GridLogMessage << "[OpenGaugeBoundary] type: openQCD"
      << std::endl;
    s << GridLogMessage << "[OpenGaugeBoundary] time direction: "
      << parameters_.timeDir << std::endl;
    s << GridLogMessage << "[OpenGaugeBoundary] cG: "
      << parameters_.cG << std::endl;
    return s.str();
  }

  // Set the unused forward temporal link at x0=T to zero in a copy.  This is
  // useful for compatibility/reference checks with the original prototype;
  // the weighted action and force below do not rely on non-group-valued stored
  // links.
  void boundaryField(const GaugeField &U, GaugeField &Ub) const {
    Ub = U;
    if (!isOpen()) return;

    GridBase *grid = U.Grid();
    const int t = parameters_.timeDir;
    const int T = grid->GlobalDimensions()[t] - 1;
    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    GaugeLinkField Ut = PeekIndex<LorentzIndex>(U, t);
    GaugeLinkField zero(grid); zero = Zero();
    Ut = where(x0 == Integer(T), zero, Ut);
    PokeIndex<LorentzIndex>(Ub, Ut, t);
  }

  ComplexField loopWeight(GridBase *grid, const std::vector<int> &path) const {
    ComplexField weight(grid);
    weight = 1.0;
    if (!isOpen()) return weight;

    const int t = parameters_.timeDir;
    const int T = grid->GlobalDimensions()[t] - 1;
    GRID_ASSERT(T >= 1);

    int displacement = 0;
    int minimum = 0;
    int maximum = 0;
    bool spatial = true;
    for (auto step : path) {
      int mu = std::abs(step) - 1;
      GRID_ASSERT(mu >= 0 && mu < Nd);
      if (mu == t) {
        spatial = false;
        displacement += (step > 0) ? 1 : -1;
        minimum = std::min(minimum, displacement);
        maximum = std::max(maximum, displacement);
      }
    }
    GRID_ASSERT(displacement == 0);

    LatticeInteger x0(grid);
    LatticeCoordinate(x0, t);
    ComplexField zero(grid); zero = 0.0;

    // A translated path is present only if every vertex lies in [0,T].
    if (minimum < 0)
      weight = where(x0 < Integer(-minimum), zero, weight);
    if (maximum > 0)
      weight = where(x0 > Integer(T - maximum), zero, weight);

    // CLS boundary improvement: every purely spatial loop on x0=0 or T
    // carries cG/2.  Bulk loops carry weight one.
    if (spatial) {
      ComplexField boundaryWeight(grid);
      boundaryWeight = parameters_.cG * 0.5;
      weight = where((x0 == Integer(0)) || (x0 == Integer(T)),
                     boundaryWeight, weight);
    }
    return weight;
  }

  RealD action(const GaugeField &U, const std::vector<Loop> &loops) const {
    std::vector<GaugeLinkField> links = extractLinks(U);
    RealD result = 0.0;
    for (const auto &loop : loops) {
      GaugeLinkField transporter(U.Grid());
      pathProduct(transporter, links, loop.path, Coordinate(Nd, 0));
      ComplexField weight = loopWeight(U.Grid(), loop.path);
      ComplexField density(U.Grid());
      density = weight *
                (RealD(Gimpl::num_colours) - real(trace(transporter)));
      result += loop.coefficient * real(TensorRemove(sum(density))) /
                RealD(Gimpl::num_colours);
    }
    return result;
  }

  void derivative(const GaugeField &U, const std::vector<Loop> &loops,
                  GaugeField &dSdU) const {
    GridBase *grid = U.Grid();
    std::vector<GaugeLinkField> links = extractLinks(U);
    std::vector<GaugeLinkField> staples(Nd, grid);
    for (int mu = 0; mu < Nd; ++mu) staples[mu] = Zero();

    for (const auto &loop : loops) {
      ComplexField baseWeight = loopWeight(grid, loop.path);
      accumulateLoopStaples(links, loop.path, baseWeight,
                            loop.coefficient, staples);
    }

    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField force =
          Ta(links[mu] * staples[mu]) *
          (0.5 / RealD(Gimpl::num_colours));
      PokeIndex<LorentzIndex>(dSdU, force, mu);
    }
  }

  static std::vector<Loop> plaquetteLoops(RealD coefficient) {
    std::vector<Loop> loops;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        loops.emplace_back(
            std::vector<int>{mu + 1, nu + 1, -(mu + 1), -(nu + 1)},
            coefficient, "plaquette");
      }
    }
    return loops;
  }

  static std::vector<Loop> plaquetteRectangleLoops(RealD c_plaq,
                                                    RealD c_rect) {
    std::vector<Loop> loops = plaquetteLoops(c_plaq);
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        loops.emplace_back(
            std::vector<int>{mu + 1, mu + 1, nu + 1,
                             -(mu + 1), -(mu + 1), -(nu + 1)},
            c_rect, "rectangle");
        loops.emplace_back(
            std::vector<int>{mu + 1, nu + 1, nu + 1,
                             -(mu + 1), -(nu + 1), -(nu + 1)},
            c_rect, "rectangle");
      }
    }
    return loops;
  }

private:
  OpenGaugeBoundaryParameters parameters_;

  static std::vector<GaugeLinkField> extractLinks(const GaugeField &U) {
    std::vector<GaugeLinkField> links(Nd, U.Grid());
    for (int mu = 0; mu < Nd; ++mu)
      links[mu] = PeekIndex<LorentzIndex>(U, mu);
    return links;
  }

  static GaugeLinkField shiftGauge(const GaugeLinkField &in,
                                   const Coordinate &shift) {
    GaugeLinkField out = in;
    for (int mu = 0; mu < Nd; ++mu) {
      int n = shift[mu];
      int direction = (n >= 0) ? 1 : -1;
      for (int i = 0; i < std::abs(n); ++i)
        out = Gimpl::CshiftLink(out, mu, direction);
    }
    return out;
  }

  static ComplexField shiftScalar(const ComplexField &in,
                                 const Coordinate &shift) {
    ComplexField out = in;
    for (int mu = 0; mu < Nd; ++mu) {
      int n = shift[mu];
      int direction = (n >= 0) ? 1 : -1;
      for (int i = 0; i < std::abs(n); ++i)
        out = Cshift(out, mu, direction);
    }
    return out;
  }

  static void pathProduct(GaugeLinkField &out,
                          const std::vector<GaugeLinkField> &links,
                          const std::vector<int> &path,
                          Coordinate displacement) {
    out = 1.0;
    for (auto step : path) {
      const int mu = std::abs(step) - 1;
      if (step > 0) {
        out = out * shiftGauge(links[mu], displacement);
        displacement[mu]++;
      } else {
        displacement[mu]--;
        out = out * shiftGauge(adj(links[mu]), displacement);
      }
    }
  }

  static void accumulateLoopStaples(
      const std::vector<GaugeLinkField> &links,
      const std::vector<int> &path,
      const ComplexField &baseWeight,
      RealD coefficient,
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
        for (int j = 1; j < n; ++j)
          rest.push_back(path[(i + j) % n]);
      } else {
        target[mu]--;
        // Reverse the closed path, starting with the inverse of this
        // backward step.  The remaining reversed steps form the staple.
        for (int j = 1; j < n; ++j) {
          int index = (i - j + n) % n;
          rest.push_back(-path[index]);
        }
      }

      Coordinate initial(Nd, 0);
      initial[mu] = 1;
      GaugeLinkField staple(baseWeight.Grid());
      pathProduct(staple, links, rest, initial);

      Coordinate weightShift(Nd, 0);
      for (int d = 0; d < Nd; ++d) weightShift[d] = -target[d];
      ComplexField linkWeight = shiftScalar(baseWeight, weightShift);
      staples[mu] = staples[mu] + coefficient * linkWeight * staple;

      before[mu] += (step > 0) ? 1 : -1;
    }

    for (int d = 0; d < Nd; ++d) GRID_ASSERT(before[d] == 0);
  }
};

NAMESPACE_END(Grid);

#endif
