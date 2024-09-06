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

#pragma once

#include <igl/MeshBooleanType.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace atek::utils {

/**
 * Calls the igl library to compute the union of two trimesh meshes.
 * @param trimeshA A Trimesh PyObject representing the first mesh.
 * @param trimeshB A Trimesh PyObject representing the second mesh.
 * @return A Trimesh PyObject representing the new mesh after union.
 */
py::object unionMeshes(const py::object& trimeshA, const py::object& trimeshB);

/**
 * Calls the Python trimesh library to compute the intersection of two meshes.
 * @param trimeshA A Trimesh PyObject representing the first mesh.
 * @param trimeshB A Trimesh hPyObject representing the second mesh.
 * @return A Trimesh PyObject representing the new mesh after intersection.
 */
py::object intersectMeshes(
    const py::object& trimeshA,
    const py::object& trimeshB);

/**
 * Calls the Python trimesh library to compute the boolean of two meshes.
 * @param trimeshA A Trimesh PyObject representing the first mesh.
 * @param trimeshB A Trimesh hPyObject representing the second mesh.
 * @param mode The type of boolean operation to perform. See MeshBooleanType.
 * @return A Trimesh PyObject representing the new mesh after intersection.
 */
py::object booleanMeshes(
    const py::object& trimeshA,
    const py::object& trimeshB,
    const igl::MeshBooleanType mode);

// Initialize the Python interpreter and keep it alive
PYBIND11_MODULE(mesh_boolean_utils, m) {
  m.def("union_meshes", &unionMeshes);
  m.def("intersect_meshes", &intersectMeshes);
  m.def("boolean_meshes", &booleanMeshes);
  py::enum_<igl::MeshBooleanType>(m, "MeshBooleanType")
      .value("UNION", igl::MeshBooleanType::MESH_BOOLEAN_TYPE_UNION)
      .value("INTERSECT", igl::MeshBooleanType::MESH_BOOLEAN_TYPE_INTERSECT)
      .value("MINUS", igl::MeshBooleanType::MESH_BOOLEAN_TYPE_MINUS)
      .value("XOR", igl::MeshBooleanType::MESH_BOOLEAN_TYPE_XOR)
      .value("RESOLVE", igl::MeshBooleanType::MESH_BOOLEAN_TYPE_RESOLVE)
      .value(
          "NUM_MESH_BOOLEAN_TYPES",
          igl::MeshBooleanType::NUM_MESH_BOOLEAN_TYPES)
      .export_values();
}

} // namespace atek::utils
