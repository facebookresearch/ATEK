#pragma once

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

// Initialize the Python interpreter and keep it alive
PYBIND11_MODULE(mesh_boolean_utils, m) {
  m.def("union_meshes", &unionMeshes);
  m.def("intersect_meshes", &intersectMeshes);
}

} // namespace atek::utils
