#include "../../include/RI/global/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <sstream>
namespace py = pybind11;

void test_list(const std::initializer_list<std::size_t>& v) {
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
PYBIND11_MODULE(PyLRI, m) {
    m.doc() = "Python binding for LRI";
    m.def("test_list", &test_list);
    py::class_<RI::Shape_Vector>(m, "Shape_Vector")
        // .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())

        .def(py::init<const RI::Shape_Vector&>())
        .def(py::init<RI::Shape_Vector&>(), py::return_value_policy::move)
        // .def_readonly("v", &RI::Shape_Vector::v)
        .def("__repr__", [](const RI::Shape_Vector& self) {
        std::stringstream ss;
        ss << "[";
        for (int i = 0;i < self.size();++i) { ss << self[i] << " "; }
        ss << "]";
        return ss.str();})
        .def("begin", &RI::Shape_Vector::begin)
        .def("end", &RI::Shape_Vector::end)
        .def("size", &RI::Shape_Vector::size)
        .def("empty", &RI::Shape_Vector::empty)
        // operator[]
        .def("__getitem__", [](const RI::Shape_Vector& v, std::size_t i) { return v[i]; })
        .def("__setitem__", [](RI::Shape_Vector& v, std::size_t i) -> std::size_t& { return v[i]; })
        // operator=
        .def("__copy__", [](RI::Shape_Vector& self, const RI::Shape_Vector& v) -> RI::Shape_Vector& { return self = v; })
        // if the second argument is  RI::Shape_Vector& v, it will lead to compile error
        .def("__assign__", [](RI::Shape_Vector& self, RI::Shape_Vector& v) -> RI::Shape_Vector& { return self = std::move(v); }, py::return_value_policy::move);
    // initializer_list seems not supported in pybind11.
    // .def(py::init<const std::initializer_list<std::size_t>&>())
    // .def(py::init<const std::initializer_list<std::size_t>&>(), [](const py::list& l) {
    // std::vector<std::size_t> v;
    // for (auto i : l) {
    //     v.push_back(i.cast<std::size_t>());
    // }
    // if (v.size() == 1) {
    //     return RI::Shape_Vector({ v[0] });
    // }
    // else if (v.size() == 2) {
    //     return RI::Shape_Vector({ v[0], v[1] });
    // }
    // else if (v.size() == 3) {
    //     return RI::Shape_Vector({ v[0], v[1], v[2] });
    // }
    // else if (v.size() == 4) {
    //     return RI::Shape_Vector({ v[0], v[1], v[2], v[3] });
    // }
    // else {
    //     throw std::invalid_argument("The size of the list must be less than or equal to 4");
    // }
    //     })
    py::class_<RI::Tensor<double>>(m, "Tensor")
        .def(py::init<const RI::Shape_Vector&>())
        // .def(py::init<const RI::Shape_Vector&, std::shared_ptr<std::valarray<double>>>())
        .def(py::init<>())
        .def(py::init<const RI::Tensor<double>&>())
        .def(py::init<RI::Tensor<double>&>(), py::return_value_policy::move)
        .def("__copy__", [](RI::Tensor<double>& self, const RI::Tensor<double>& v) { return self = v; })
        .def("__assign__", [](RI::Tensor<double>& self, RI::Tensor<double>& v) { return self = std::move(v); }, py::return_value_policy::move)
        .def("get_shape_all", &RI::Tensor<double>::get_shape_all)
        .def("reshape", &RI::Tensor<double>::reshape)
        .def("copy", &RI::Tensor<double>::copy)
        .def("transpose", &RI::Tensor<double>::transpose)
        // .def("norm", &RI::Tensor<double>::norm) // depending on blas
        // .def("ptr", &RI::Tensor<double>::ptr)    // pybind11 auto-converts double* to a float...
        // to bind a function that returns pointer, we need to manually convert it to a py::array_t
        .def("ptr", [](RI::Tensor<double>& self) {
        return py::array_t<double>(self.get_shape_all(), self.ptr(), py::cast(self));   // py::cast(self) ensure the data not copied and modifications are reflected back in the C++ object
            }, py::return_value_policy::reference_internal) //the lifetime of the C++ object is tied to the lifetime of the numpy array
        .def("empty", &RI::Tensor<double>::empty)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def("__call__", [](const RI::Tensor<double>& t, std::size_t i0) { return t(i0); })
        .def("__call__", [](const RI::Tensor<double>& t, std::size_t i0, std::size_t i1) { return t(i0, i1); })
        .def("__call__", [](const RI::Tensor<double>& t, std::size_t i0, std::size_t i1, std::size_t i2) { return t(i0, i1, i2); })
        .def("__call__", [](const RI::Tensor<double>& t, std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) { return t(i0, i1, i2, i3); });
}