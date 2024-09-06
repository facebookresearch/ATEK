/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mesh_boolean_utils.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <igl/copyleft/cgal/mesh_boolean.h>
#include <Eigen/Core>

#include <utility>

namespace py = pybind11;

namespace {
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> convertTrimeshToEigen(
    const py::object& trimesh) {
  py::array vertices_array = trimesh.attr("vertices").cast<py::array>();
  Eigen::MatrixXd V = vertices_array.cast<Eigen::MatrixXd>();

  py::array faces_array = trimesh.attr("faces").cast<py::array>();
  Eigen::MatrixXi F = faces_array.cast<Eigen::MatrixXi>();

  return std::make_pair(V, F);
}

py::object convertEigenToTrimesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F) {
  auto trimeshClass = py::module_::import("trimesh").attr("Trimesh");

  py::array V_array = py::cast(V);
  py::array F_array = py::cast(F);
  py::object mesh_obj = trimeshClass(V_array, F_array);
  return mesh_obj;
}

} // namespace

namespace atek::utils {

py::object unionMeshes(const py::object& trimeshA, const py::object& trimeshB) {
  const auto [VA, FA] = convertTrimeshToEigen(trimeshA);
  const auto [VB, FB] = convertTrimeshToEigen(trimeshB);

  Eigen::MatrixXd VU;
  Eigen::MatrixXi FU;
  igl::copyleft::cgal::mesh_boolean(
      VA, FA, VB, FB, igl::MESH_BOOLEAN_TYPE_UNION, VU, FU);

  return convertEigenToTrimesh(VU, FU);
}

py::object intersectMeshes(
    const py::object& trimeshA,
    const py::object& trimeshB) {
  const auto [VA, FA] = convertTrimeshToEigen(trimeshA);
  const auto [VB, FB] = convertTrimeshToEigen(trimeshB);

  Eigen::MatrixXd VI;
  Eigen::MatrixXi FI;
  igl::copyleft::cgal::mesh_boolean(
      VA, FA, VB, FB, igl::MESH_BOOLEAN_TYPE_INTERSECT, VI, FI);

  return convertEigenToTrimesh(VI, FI);
}

py::object booleanMeshes(
    const py::object& trimeshA,
    const py::object& trimeshB,
    const igl::MeshBooleanType mode) {
  const auto [VA, FA] = convertTrimeshToEigen(trimeshA);
  const auto [VB, FB] = convertTrimeshToEigen(trimeshB);

  Eigen::MatrixXd VI;
  Eigen::MatrixXi FI;
  igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, mode, VI, FI);

  return convertEigenToTrimesh(VI, FI);
}

} // namespace atek::utils
;
