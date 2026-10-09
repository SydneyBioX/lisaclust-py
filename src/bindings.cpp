// Python bindings for the lisaClust C++ core (src/core, synced from lisaClust; never edited here).
// They mirror lisaClust's R bindings (src/bindings.cpp there) one for one: convert, call the core, return.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "lisaclust/core.hpp"

namespace py = pybind11;

namespace {

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using DoubleArrayF = py::array_t<double, py::array::f_style | py::array::forcecast>;

std::vector<lisaclust::Ring> toRings(const std::vector<DoubleArray>& rings) {
  std::vector<lisaclust::Ring> win;
  for (const DoubleArray& r : rings) {
    if (r.ndim() != 2 || r.shape(1) != 2) throw std::invalid_argument("each ring must be an (m, 2) array");
    lisaclust::Ring ring;
    for (py::ssize_t i = 0; i < r.shape(0); ++i) ring.push_back({r.at(i, 0), r.at(i, 1)});
    win.push_back(ring);
  }
  return win;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "lisaClust's C++ core";

  m.def("disc_window_area", [](DoubleArray x, DoubleArray y, double r, int npoly, std::vector<DoubleArray> rings) {
    py::array_t<double> out(x.size());
    lisaclust::discAreas(x.data(), y.data(), x.size(), r, npoly, toRings(rings), out.mutable_data());
    return out;
  });

  m.def("distance_to_boundary", [](DoubleArray x, DoubleArray y, std::vector<DoubleArray> rings) {
    py::array_t<double> out(x.size());
    lisaclust::distanceToBoundary(x.data(), y.data(), x.size(), toRings(rings), out.mutable_data());
    return out;
  });

  m.def("convex_hull", [](DoubleArray x, DoubleArray y) {
    lisaclust::Ring h = lisaclust::convexHull(x.data(), y.data(), x.size());
    py::array_t<double> out({static_cast<py::ssize_t>(h.size()), static_cast<py::ssize_t>(2)});
    auto o = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < h.size(); ++i) { o(i, 0) = h[i].x; o(i, 1) = h[i].y; }
    return out;
  });

  // type is 0-based; edge is n x (len(Rs) - 1). Returns value with shape (n, K, nb) and the presence flags.
  m.def("local_curves", [](DoubleArray x, DoubleArray y, IntArray type, int n_types, std::vector<double> Rs,
                           std::vector<double> label_val, DoubleArray wt, std::vector<double> lam, DoubleArrayF edge,
                           bool l_function, bool include_self) {
    const int n = static_cast<int>(x.size());
    if (edge.ndim() != 2 || edge.shape(0) != n || edge.shape(1) != static_cast<py::ssize_t>(Rs.size()) - 1)
      throw std::invalid_argument("edge must be an (n, len(Rs) - 1) array");
    lisaclust::LocalCurves res = lisaclust::localCurves(x.data(), y.data(), type.data(), n, n_types, Rs, label_val,
                                                        wt.data(), lam, edge.data(), l_function, include_self);
    // the core's layout, (k * K + J) * n + i, is a Fortran-order (n, K, nb) array
    py::array_t<double, py::array::f_style> value({static_cast<py::ssize_t>(res.n), static_cast<py::ssize_t>(res.K),
                                                   static_cast<py::ssize_t>(res.nb)});
    std::copy(res.value.begin(), res.value.end(), value.mutable_data());
    auto flags = [](const std::vector<char>& v) {
      py::array_t<bool> a(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) a.mutable_data()[i] = v[i] != 0;
      return a;
    };
    py::dict d;
    d["value"] = value;
    d["cell"] = flags(res.cellPresent);
    d["bin"] = flags(res.binPresent);
    d["type"] = flags(res.typePresent);
    return d;
  }, py::arg("x"), py::arg("y"), py::arg("type"), py::arg("n_types"), py::arg("Rs"), py::arg("label_val"),
     py::arg("wt"), py::arg("lam"), py::arg("edge"), py::arg("l_function"), py::arg("include_self") = true);

  // labels are 0-based; -1 when there are no training points
  m.def("nearest_labels", [](DoubleArray tx, DoubleArray ty, IntArray label, DoubleArray qx, DoubleArray qy) {
    std::vector<int> res = lisaclust::nearestLabels(tx.data(), ty.data(), label.data(), static_cast<int>(tx.size()),
                                                    qx.data(), qy.data(), static_cast<int>(qx.size()));
    py::array_t<int> out(res.size());
    std::copy(res.begin(), res.end(), out.mutable_data());
    return out;
  });
}
