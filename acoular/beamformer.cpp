#ifdef __CPLUSPLUS__
extern "C" {
#endif

#ifndef __GNUC__
#pragma warning(disable: 4275)
#pragma warning(disable: 4101)

#endif
#include "Python.h"
#include "blitz/array.h"
#include "compile.h"
#include "frameobject.h"
#include <complex>
#include <math.h>
#include <string>
#include "scxx/object.h"
#include "scxx/list.h"
#include "scxx/tuple.h"
#include "scxx/dict.h"
#include <iostream>
#include <stdio.h>
#include "numpy/arrayobject.h"




// global None value for use in functions.
namespace py {
object None = object(Py_None);
}

const char* find_type(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    if(PyFile_Check(py_obj)) return "file";
    if(PyModule_Check(py_obj)) return "module";

    //should probably do more intergation (and thinking) on these.
    if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance";
    if(PyCallable_Check(py_obj)) return "callable";
    return "unknown type";
}

void throw_error(PyObject* exc, const char* msg)
{
 //printf("setting python error: %s\n",msg);
  PyErr_SetString(exc, msg);
  //printf("throwing error\n");
  throw 1;
}

void handle_bad_type(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}

void handle_conversion_error(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}


class int_handler
{
public:
    int convert_to_int(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInt_Check(py_obj))
            handle_conversion_error(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }

    int py_to_int(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInt_Check(py_obj))
            handle_bad_type(py_obj,"int", name);
        
        return (int) PyInt_AsLong(py_obj);
    }
};

int_handler x__int_handler = int_handler();
#define convert_to_int(py_obj,name) \
        x__int_handler.convert_to_int(py_obj,name)
#define py_to_int(py_obj,name) \
        x__int_handler.py_to_int(py_obj,name)


PyObject* int_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class float_handler
{
public:
    double convert_to_float(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_conversion_error(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }

    double py_to_float(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_bad_type(py_obj,"float", name);
        
        return PyFloat_AsDouble(py_obj);
    }
};

float_handler x__float_handler = float_handler();
#define convert_to_float(py_obj,name) \
        x__float_handler.convert_to_float(py_obj,name)
#define py_to_float(py_obj,name) \
        x__float_handler.py_to_float(py_obj,name)


PyObject* float_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class complex_handler
{
public:
    std::complex<double> convert_to_complex(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_conversion_error(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }

    std::complex<double> py_to_complex(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_bad_type(py_obj,"complex", name);
        
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }
};

complex_handler x__complex_handler = complex_handler();
#define convert_to_complex(py_obj,name) \
        x__complex_handler.convert_to_complex(py_obj,name)
#define py_to_complex(py_obj,name) \
        x__complex_handler.py_to_complex(py_obj,name)


PyObject* complex_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class unicode_handler
{
public:
    Py_UNICODE* convert_to_unicode(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_conversion_error(py_obj,"unicode", name);
        return PyUnicode_AS_UNICODE(py_obj);
    }

    Py_UNICODE* py_to_unicode(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_bad_type(py_obj,"unicode", name);
        Py_XINCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);
    }
};

unicode_handler x__unicode_handler = unicode_handler();
#define convert_to_unicode(py_obj,name) \
        x__unicode_handler.convert_to_unicode(py_obj,name)
#define py_to_unicode(py_obj,name) \
        x__unicode_handler.py_to_unicode(py_obj,name)


PyObject* unicode_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class string_handler
{
public:
    std::string convert_to_string(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return std::string(PyString_AsString(py_obj));
    }

    std::string py_to_string(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyString_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        Py_XINCREF(py_obj);
        return std::string(PyString_AsString(py_obj));
    }
};

string_handler x__string_handler = string_handler();
#define convert_to_string(py_obj,name) \
        x__string_handler.convert_to_string(py_obj,name)
#define py_to_string(py_obj,name) \
        x__string_handler.py_to_string(py_obj,name)


               PyObject* string_to_py(std::string s)
               {
                   return PyString_FromString(s.c_str());
               }
               
class list_handler
{
public:
    py::list convert_to_list(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return py::list(py_obj);
    }

    py::list py_to_list(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyList_Check(py_obj))
            handle_bad_type(py_obj,"list", name);
        
        return py::list(py_obj);
    }
};

list_handler x__list_handler = list_handler();
#define convert_to_list(py_obj,name) \
        x__list_handler.convert_to_list(py_obj,name)
#define py_to_list(py_obj,name) \
        x__list_handler.py_to_list(py_obj,name)


PyObject* list_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class dict_handler
{
public:
    py::dict convert_to_dict(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return py::dict(py_obj);
    }

    py::dict py_to_dict(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyDict_Check(py_obj))
            handle_bad_type(py_obj,"dict", name);
        
        return py::dict(py_obj);
    }
};

dict_handler x__dict_handler = dict_handler();
#define convert_to_dict(py_obj,name) \
        x__dict_handler.convert_to_dict(py_obj,name)
#define py_to_dict(py_obj,name) \
        x__dict_handler.py_to_dict(py_obj,name)


PyObject* dict_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class tuple_handler
{
public:
    py::tuple convert_to_tuple(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return py::tuple(py_obj);
    }

    py::tuple py_to_tuple(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_bad_type(py_obj,"tuple", name);
        
        return py::tuple(py_obj);
    }
};

tuple_handler x__tuple_handler = tuple_handler();
#define convert_to_tuple(py_obj,name) \
        x__tuple_handler.convert_to_tuple(py_obj,name)
#define py_to_tuple(py_obj,name) \
        x__tuple_handler.py_to_tuple(py_obj,name)


PyObject* tuple_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class file_handler
{
public:
    FILE* convert_to_file(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"file", name);
        return PyFile_AsFile(py_obj);
    }

    FILE* py_to_file(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"file", name);
        Py_XINCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
};

file_handler x__file_handler = file_handler();
#define convert_to_file(py_obj,name) \
        x__file_handler.convert_to_file(py_obj,name)
#define py_to_file(py_obj,name) \
        x__file_handler.py_to_file(py_obj,name)


               PyObject* file_to_py(FILE* file, const char* name,
                                    const char* mode)
               {
                   return (PyObject*) PyFile_FromFile(file,
                     const_cast<char*>(name),
                     const_cast<char*>(mode), fclose);
               }
               
class instance_handler
{
public:
    py::object convert_to_instance(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_conversion_error(py_obj,"instance", name);
        return py::object(py_obj);
    }

    py::object py_to_instance(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_bad_type(py_obj,"instance", name);
        
        return py::object(py_obj);
    }
};

instance_handler x__instance_handler = instance_handler();
#define convert_to_instance(py_obj,name) \
        x__instance_handler.convert_to_instance(py_obj,name)
#define py_to_instance(py_obj,name) \
        x__instance_handler.py_to_instance(py_obj,name)


PyObject* instance_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class numpy_size_handler
{
public:
    void conversion_numpy_check_size(PyArrayObject* arr_obj, int Ndims,
                                     const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"Conversion Error: received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_size(PyArrayObject* arr_obj, int Ndims, const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_size_handler x__numpy_size_handler = numpy_size_handler();
#define conversion_numpy_check_size x__numpy_size_handler.conversion_numpy_check_size
#define numpy_check_size x__numpy_size_handler.numpy_check_size


class numpy_type_handler
{
public:
    void conversion_numpy_check_type(PyArrayObject* arr_obj, int numeric_type,
                                     const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {

        const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                "float", "double", "longdouble", "cfloat", "cdouble",
                                "clongdouble", "object", "string", "unicode", "void", "ntype",
                                "unknown"};
        char msg[500];
        sprintf(msg,"Conversion Error: received '%s' typed array instead of '%s' typed array for variable '%s'",
                type_names[arr_type],type_names[numeric_type],name);
        throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_type(PyArrayObject* arr_obj, int numeric_type, const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {
            const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                    "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                    "float", "double", "longdouble", "cfloat", "cdouble",
                                    "clongdouble", "object", "string", "unicode", "void", "ntype",
                                    "unknown"};
            char msg[500];
            sprintf(msg,"received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_type_handler x__numpy_type_handler = numpy_type_handler();
#define conversion_numpy_check_type x__numpy_type_handler.conversion_numpy_check_type
#define numpy_check_type x__numpy_type_handler.numpy_check_type


class numpy_handler
{
public:
    PyArrayObject* convert_to_numpy(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyArray_Check(py_obj))
            handle_conversion_error(py_obj,"numpy", name);
        return (PyArrayObject*) py_obj;
    }

    PyArrayObject* py_to_numpy(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyArray_Check(py_obj))
            handle_bad_type(py_obj,"numpy", name);
        Py_XINCREF(py_obj);
        return (PyArrayObject*) py_obj;
    }
};

numpy_handler x__numpy_handler = numpy_handler();
#define convert_to_numpy(py_obj,name) \
        x__numpy_handler.convert_to_numpy(py_obj,name)
#define py_to_numpy(py_obj,name) \
        x__numpy_handler.py_to_numpy(py_obj,name)


PyObject* numpy_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class catchall_handler
{
public:
    py::object convert_to_catchall(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !(py_obj))
            handle_conversion_error(py_obj,"catchall", name);
        return py::object(py_obj);
    }

    py::object py_to_catchall(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !(py_obj))
            handle_bad_type(py_obj,"catchall", name);
        
        return py::object(py_obj);
    }
};

catchall_handler x__catchall_handler = catchall_handler();
#define convert_to_catchall(py_obj,name) \
        x__catchall_handler.convert_to_catchall(py_obj,name)
#define py_to_catchall(py_obj,name) \
        x__catchall_handler.py_to_catchall(py_obj,name)


PyObject* catchall_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}



// This should be declared only if they are used by some function
// to keep from generating needless warnings. for now, we'll always
// declare them.

int _beg = blitz::fromStart;
const int _end = blitz::toEnd;
blitz::Range _all = blitz::Range::all();

template<class T, int N>
static blitz::Array<T,N> convert_to_blitz(PyArrayObject* arr_obj,const char* name)
{
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyArrayObject* arr_obj,const char* name)
{

    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}


static PyObject* faverage(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","ft","local_dict", NULL};
    PyObject *py_csm, *py_ft;
    int csm_used, ft_used;
    py_csm = py_ft = NULL;
    csm_used= ft_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OO|O:faverage",const_cast<char**>(kwlist),&py_csm, &py_ft, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_ft = py_ft;
        PyArrayObject* ft_array = convert_to_numpy(py_ft,"ft");
        conversion_numpy_check_type(ft_array,PyArray_CDOUBLE,"ft");
        conversion_numpy_check_size(ft_array,2,"ft");
        blitz::Array<std::complex<double> ,2> ft = convert_to_blitz<std::complex<double> ,2>(ft_array,"ft");
        blitz::TinyVector<int,2> Nft = ft.shape();
        ft_used = 1;
        /*<function call here>*/     
         
            std::complex<double> temp;
            int nf=Ncsm[0]; 
            int nc=Ncsm[1];
            int f,i,j;
        #pragma omp parallel private(f,i,j,temp) shared(csm,nc,nf,ft)
            {
        #pragma omp for schedule(auto) nowait 
            for (f=0; f<nf; ++f) {
                for (i=0; i<nc; ++i) {
                    temp=conj(ft(f,i));
                    for (j=0; j<nc; ++j) {
                         csm(f,i,j)+=temp * ft(f,j);
                    }
                }
            }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(ft_used)
    {
        Py_XDECREF(py_ft);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamdiag",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=r01/nc;
                    rs*=rs;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamfull",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=r01/nc;
                    rs*=rs;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                        temp1+=(csm(i,ii,ii)*conj(e1[ii])*e1[ii]).real();
            
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_3d(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamdiag_3d",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=1.0/nc;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_3d(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamfull_3d",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=1.0/nc;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                        temp1+=(csm(i,ii,ii)*conj(e1[ii])*e1[ii]).real();
            
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_classic(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamdiag_classic",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2;
                    }
                    rs=1.0;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_classic(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamfull_classic",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2;
                    }
                    rs=1.0;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                        temp1+=(csm(i,ii,ii)*conj(e1[ii])*e1[ii]).real();
            
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_inverse(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamdiag_inverse",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2*rm1;
                    }
                    rs=r01;
                    rs*=rs;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_inverse(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"csm","e","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_csm, *py_e, *py_h, *py_r0, *py_rm, *py_kj;
    int csm_used, e_used, h_used, r0_used, rm_used, kj_used;
    py_csm = py_e = py_h = py_r0 = py_rm = py_kj = NULL;
    csm_used= e_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beamfull_inverse",const_cast<char**>(kwlist),&py_csm, &py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_csm = py_csm;
        PyArrayObject* csm_array = convert_to_numpy(py_csm,"csm");
        conversion_numpy_check_type(csm_array,PyArray_CDOUBLE,"csm");
        conversion_numpy_check_size(csm_array,3,"csm");
        blitz::Array<std::complex<double> ,3> csm = convert_to_blitz<std::complex<double> ,3>(csm_array,"csm");
        blitz::TinyVector<int,3> Ncsm = csm.shape();
        csm_used = 1;
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2;
            std::complex<double>* temp4;
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double temp1,rs,r01,rm1,kjj;
            float temp3;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,jj;
        #pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp3=(float)(kjj*(r01-rm1));
                        temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                        
                        e1[ii]=temp2*rm1;
                    }
                    rs=r01;
                    rs*=rs;
            
                    temp1=0.0; 
                    for (ii=0; ii<nc; ++ii) {
                        temp2=0.0;
                        temp4=&csm(i,ii);
                        for (jj=0; jj<ii; ++jj) {
                            temp2+=(*(temp4++))*(e1[jj]);
                        }
                        temp1+=2*(temp2*conj(e1[ii])).real();
        //                printf("%d %d %d %d %f\n",omp_get_thread_num(),p,ii,jj,temp1);
                        
                        temp1+=(csm(i,ii,ii)*conj(e1[ii])*e1[ii]).real();
            
                    }
                    h(i,p)=temp1/rs;
                }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(csm_used)
    {
        Py_XDECREF(py_csm);
    }
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_os(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamdiag_os",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=r01/nc;
                    rs*=rs;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2)-temp1)*eva(i,nn)).real();   
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_os(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamfull_os",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=r01/nc;
                    rs*=rs;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2))*eva(i,nn)).real();
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_os_3d(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamdiag_os_3d",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=1.0/nc;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2)-temp1)*eva(i,nn)).real();   
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_os_3d(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamfull_os_3d",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2/rm1;
                    }
                    rs*=1.0/nc;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2))*eva(i,nn)).real();
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_os_classic(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamdiag_os_classic",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2;
                    }
                    rs=1.0;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2)-temp1)*eva(i,nn)).real();   
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_os_classic(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamfull_os_classic",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2;
                    }
                    rs=1.0;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2))*eva(i,nn)).real();
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamdiag_os_inverse(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamdiag_os_inverse",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2*rm1;
                    }
                    rs=r01;
                    rs*=rs;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2)-temp1)*eva(i,nn)).real();   
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beamfull_os_inverse(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","h","r0","rm","kj","eva","eve","nmin","nmax","local_dict", NULL};
    PyObject *py_e, *py_h, *py_r0, *py_rm, *py_kj, *py_eva, *py_eve, *py_nmin, *py_nmax;
    int e_used, h_used, r0_used, rm_used, kj_used, eva_used, eve_used, nmin_used, nmax_used;
    py_e = py_h = py_r0 = py_rm = py_kj = py_eva = py_eve = py_nmin = py_nmax = NULL;
    e_used= h_used= r0_used= rm_used= kj_used= eva_used= eve_used= nmin_used= nmax_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOOOO|O:r_beamfull_os_inverse",const_cast<char**>(kwlist),&py_e, &py_h, &py_r0, &py_rm, &py_kj, &py_eva, &py_eve, &py_nmin, &py_nmax, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        py_eva = py_eva;
        PyArrayObject* eva_array = convert_to_numpy(py_eva,"eva");
        conversion_numpy_check_type(eva_array,PyArray_DOUBLE,"eva");
        conversion_numpy_check_size(eva_array,2,"eva");
        blitz::Array<double,2> eva = convert_to_blitz<double,2>(eva_array,"eva");
        blitz::TinyVector<int,2> Neva = eva.shape();
        eva_used = 1;
        py_eve = py_eve;
        PyArrayObject* eve_array = convert_to_numpy(py_eve,"eve");
        conversion_numpy_check_type(eve_array,PyArray_CDOUBLE,"eve");
        conversion_numpy_check_size(eve_array,3,"eve");
        blitz::Array<std::complex<double> ,3> eve = convert_to_blitz<std::complex<double> ,3>(eve_array,"eve");
        blitz::TinyVector<int,3> Neve = eve.shape();
        eve_used = 1;
        py_nmin = py_nmin;
        int nmin = convert_to_int(py_nmin,"nmin");
        nmin_used = 1;
        py_nmax = py_nmax;
        int nmax = convert_to_int(py_nmax,"nmax");
        nmax_used = 1;
        /*<function call here>*/     
        
            std::complex<double> temp2,temp3;
            std::complex<double>* temp5;    
            int numpoints=Nr0[0];
            int nc=Nrm[1];   
            std::complex<double> e1[nc];
            int numfreq=Nkj[0];
            double rs,r01,rm1,temp1,kjj;
            float temp4;
            if (nmin<0) {
                nmin=0;
                }
            if (nmax>nc) {
                nmax=nc;
                }
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i).imag();
                int p,ii,nn;
        #pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
                {
        #pragma omp for schedule(auto) nowait 
                for (p=0; p<numpoints; ++p) {
                    rs=0;
                    h(i,p)=0.0;
                    r01=r0(p);
                    for (ii=0; ii<nc; ++ii) {
                        rm1=rm(p,ii);
                        rs+=1.0/(rm1*rm1);
                        temp4 = (float)(kjj*(r01-rm1));
                        temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                        
                        e1[ii]=temp2*rm1;
                    }
                    rs=r01;
                    rs*=rs;
            
                    for (nn=nmin; nn<nmax; ++nn) {
                        temp2=0.0;
                        temp1=0.0;
                        temp5 = e1;
                        for (int ii=0; ii<nc; ++ii) {
                            temp3=eve(i,ii,nn)*(*(temp5++));
                            temp2+=temp3;
                            temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                        }
                        
                        h(i,p)+=((temp2*conj(temp2))*eva(i,nn)).real();
            
                    }
                    h(i,p)*=1./rs;
                }
                }
                
            }
            
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(eva_used)
    {
        Py_XDECREF(py_eva);
    }
    if(eve_used)
    {
        Py_XDECREF(py_eve);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beam_psf(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"e","f","h","r0","rm","kj","local_dict", NULL};
    PyObject *py_e, *py_f, *py_h, *py_r0, *py_rm, *py_kj;
    int e_used, f_used, h_used, r0_used, rm_used, kj_used;
    py_e = py_f = py_h = py_r0 = py_rm = py_kj = NULL;
    e_used= f_used= h_used= r0_used= rm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beam_psf",const_cast<char**>(kwlist),&py_e, &py_f, &py_h, &py_r0, &py_rm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_e = py_e;
        PyArrayObject* e_array = convert_to_numpy(py_e,"e");
        conversion_numpy_check_type(e_array,PyArray_CDOUBLE,"e");
        conversion_numpy_check_size(e_array,1,"e");
        blitz::Array<std::complex<double> ,1> e = convert_to_blitz<std::complex<double> ,1>(e_array,"e");
        blitz::TinyVector<int,1> Ne = e.shape();
        e_used = 1;
        py_f = py_f;
        PyArrayObject* f_array = convert_to_numpy(py_f,"f");
        conversion_numpy_check_type(f_array,PyArray_CDOUBLE,"f");
        conversion_numpy_check_size(f_array,1,"f");
        blitz::Array<std::complex<double> ,1> f = convert_to_blitz<std::complex<double> ,1>(f_array,"f");
        blitz::TinyVector<int,1> Nf = f.shape();
        f_used = 1;
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,3,"h");
        blitz::Array<double,3> h = convert_to_blitz<double,3>(h_array,"h");
        blitz::TinyVector<int,3> Nh = h.shape();
        h_used = 1;
        py_r0 = py_r0;
        PyArrayObject* r0_array = convert_to_numpy(py_r0,"r0");
        conversion_numpy_check_type(r0_array,PyArray_DOUBLE,"r0");
        conversion_numpy_check_size(r0_array,1,"r0");
        blitz::Array<double,1> r0 = convert_to_blitz<double,1>(r0_array,"r0");
        blitz::TinyVector<int,1> Nr0 = r0.shape();
        r0_used = 1;
        py_rm = py_rm;
        PyArrayObject* rm_array = convert_to_numpy(py_rm,"rm");
        conversion_numpy_check_type(rm_array,PyArray_DOUBLE,"rm");
        conversion_numpy_check_size(rm_array,2,"rm");
        blitz::Array<double,2> rm = convert_to_blitz<double,2>(rm_array,"rm");
        blitz::TinyVector<int,2> Nrm = rm.shape();
        rm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<float> temp1,kjj;
            int numpoints=Nrm[0];
            int nc=Nrm[1];   
            int numfreq=Nkj[0];
            float temp2;
            float r00,rmm,r0m,rs;
            for (int i=0; i<numfreq; ++i) {
                kjj=kj(i);//.imag();
                for (int j=0; j<numpoints; ++j) {
                    for (int p=0; p<numpoints; ++p) {
                        rs=0;
                        r00=r0(p);
                        temp1=0.0;
                        for (int ii=0; ii<nc; ++ii) {
                            rmm=rm(p,ii);
                            rs+=1.0/(rmm*rmm);
                            r0m=rm(j,ii);
                            temp2=(kjj*(r00+r0m-rmm)).imag();
                            e(ii)=(std::complex<double>(cosf(temp2),sinf(temp2)))*(1.0/(rmm*r0m));
                        }
                        rs*=r00/nc;
                        temp1=0.0;
                        for (int ii=0; ii<nc; ++ii) {
                            temp1+=e(ii);
                        }
                        h(i,j,p)=(temp1*conj(temp1)).real()/(rs*rs);
                    }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(e_used)
    {
        Py_XDECREF(py_e);
    }
    if(f_used)
    {
        Py_XDECREF(py_f);
    }
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(r0_used)
    {
        Py_XDECREF(py_r0);
    }
    if(rm_used)
    {
        Py_XDECREF(py_rm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beam_psf1(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"h","rt0","rs0","rtm","rsm","kj","local_dict", NULL};
    PyObject *py_h, *py_rt0, *py_rs0, *py_rtm, *py_rsm, *py_kj;
    int h_used, rt0_used, rs0_used, rtm_used, rsm_used, kj_used;
    py_h = py_rt0 = py_rs0 = py_rtm = py_rsm = py_kj = NULL;
    h_used= rt0_used= rs0_used= rtm_used= rsm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beam_psf1",const_cast<char**>(kwlist),&py_h, &py_rt0, &py_rs0, &py_rtm, &py_rsm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_rt0 = py_rt0;
        PyArrayObject* rt0_array = convert_to_numpy(py_rt0,"rt0");
        conversion_numpy_check_type(rt0_array,PyArray_DOUBLE,"rt0");
        conversion_numpy_check_size(rt0_array,1,"rt0");
        blitz::Array<double,1> rt0 = convert_to_blitz<double,1>(rt0_array,"rt0");
        blitz::TinyVector<int,1> Nrt0 = rt0.shape();
        rt0_used = 1;
        py_rs0 = py_rs0;
        PyArrayObject* rs0_array = convert_to_numpy(py_rs0,"rs0");
        conversion_numpy_check_type(rs0_array,PyArray_DOUBLE,"rs0");
        conversion_numpy_check_size(rs0_array,1,"rs0");
        blitz::Array<double,1> rs0 = convert_to_blitz<double,1>(rs0_array,"rs0");
        blitz::TinyVector<int,1> Nrs0 = rs0.shape();
        rs0_used = 1;
        py_rtm = py_rtm;
        PyArrayObject* rtm_array = convert_to_numpy(py_rtm,"rtm");
        conversion_numpy_check_type(rtm_array,PyArray_DOUBLE,"rtm");
        conversion_numpy_check_size(rtm_array,2,"rtm");
        blitz::Array<double,2> rtm = convert_to_blitz<double,2>(rtm_array,"rtm");
        blitz::TinyVector<int,2> Nrtm = rtm.shape();
        rtm_used = 1;
        py_rsm = py_rsm;
        PyArrayObject* rsm_array = convert_to_numpy(py_rsm,"rsm");
        conversion_numpy_check_type(rsm_array,PyArray_DOUBLE,"rsm");
        conversion_numpy_check_size(rsm_array,2,"rsm");
        blitz::Array<double,2> rsm = convert_to_blitz<double,2>(rsm_array,"rsm");
        blitz::TinyVector<int,2> Nrsm = rsm.shape();
        rsm_used = 1;
        py_kj = py_kj;
        std::complex<double> kj = convert_to_complex(py_kj,"kj");
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<float> term2;
            int numpoints_grid = Nrtm[0];
            int numpoints = Nrsm[0];
            int nc = Nrtm[1];    
            float expon, kj_freq;
            float r0, rsi;
        
            kj_freq = kj.imag();
        
            for (int t=0; t<numpoints_grid; ++t) {
                for (int s=0; s<numpoints; ++s) {
                    term2 = 0;
                    for (int i=0; i<nc; ++i) {
                        rsi = rsm(s,i);
                        expon = kj_freq * ( rtm(t,i) - rsi );
                        term2 += (std::complex<float>(cosf(expon),sinf(expon))) / rsi;
                    }
                    r0 = rs0(s);
                    h(t,s) = r0*r0 / (nc*nc) * (term2*conj(term2)).real();
                }
            }
         
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(rt0_used)
    {
        Py_XDECREF(py_rt0);
    }
    if(rs0_used)
    {
        Py_XDECREF(py_rs0);
    }
    if(rtm_used)
    {
        Py_XDECREF(py_rtm);
    }
    if(rsm_used)
    {
        Py_XDECREF(py_rsm);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beam_psf2(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"h","rt0","rs0","rtm","rsm","kj","local_dict", NULL};
    PyObject *py_h, *py_rt0, *py_rs0, *py_rtm, *py_rsm, *py_kj;
    int h_used, rt0_used, rs0_used, rtm_used, rsm_used, kj_used;
    py_h = py_rt0 = py_rs0 = py_rtm = py_rsm = py_kj = NULL;
    h_used= rt0_used= rs0_used= rtm_used= rsm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beam_psf2",const_cast<char**>(kwlist),&py_h, &py_rt0, &py_rs0, &py_rtm, &py_rsm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_rt0 = py_rt0;
        PyArrayObject* rt0_array = convert_to_numpy(py_rt0,"rt0");
        conversion_numpy_check_type(rt0_array,PyArray_DOUBLE,"rt0");
        conversion_numpy_check_size(rt0_array,1,"rt0");
        blitz::Array<double,1> rt0 = convert_to_blitz<double,1>(rt0_array,"rt0");
        blitz::TinyVector<int,1> Nrt0 = rt0.shape();
        rt0_used = 1;
        py_rs0 = py_rs0;
        PyArrayObject* rs0_array = convert_to_numpy(py_rs0,"rs0");
        conversion_numpy_check_type(rs0_array,PyArray_DOUBLE,"rs0");
        conversion_numpy_check_size(rs0_array,1,"rs0");
        blitz::Array<double,1> rs0 = convert_to_blitz<double,1>(rs0_array,"rs0");
        blitz::TinyVector<int,1> Nrs0 = rs0.shape();
        rs0_used = 1;
        py_rtm = py_rtm;
        PyArrayObject* rtm_array = convert_to_numpy(py_rtm,"rtm");
        conversion_numpy_check_type(rtm_array,PyArray_DOUBLE,"rtm");
        conversion_numpy_check_size(rtm_array,2,"rtm");
        blitz::Array<double,2> rtm = convert_to_blitz<double,2>(rtm_array,"rtm");
        blitz::TinyVector<int,2> Nrtm = rtm.shape();
        rtm_used = 1;
        py_rsm = py_rsm;
        PyArrayObject* rsm_array = convert_to_numpy(py_rsm,"rsm");
        conversion_numpy_check_type(rsm_array,PyArray_DOUBLE,"rsm");
        conversion_numpy_check_size(rsm_array,2,"rsm");
        blitz::Array<double,2> rsm = convert_to_blitz<double,2>(rsm_array,"rsm");
        blitz::TinyVector<int,2> Nrsm = rsm.shape();
        rsm_used = 1;
        py_kj = py_kj;
        std::complex<double> kj = convert_to_complex(py_kj,"kj");
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<float> term2;
            int numpoints_grid = Nrtm[0];
            int numpoints = Nrsm[0];
            int nc = Nrtm[1];    
            float expon, kj_freq;
            float r0, rsi, rti;
        
            kj_freq = kj.imag();
        
            for (int t=0; t<numpoints_grid; ++t) {
                for (int s=0; s<numpoints; ++s) {
                    term2 = 0;
            
                    for (int i=0; i<nc; ++i) {
                        rsi = rsm(s,i);
                        rti = rtm(t,i);
                       
                        expon = kj_freq * (rti-rsi);
                        term2 += rti/rsi * (std::complex<float>(cosf(expon),sinf(expon)));
                    }
                    r0 = rs0(s)/rt0(t);
                    h(t,s) = r0*r0 /(nc*nc) * (term2*conj(term2)).real();
                }
            }
        
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(rt0_used)
    {
        Py_XDECREF(py_rt0);
    }
    if(rs0_used)
    {
        Py_XDECREF(py_rs0);
    }
    if(rtm_used)
    {
        Py_XDECREF(py_rtm);
    }
    if(rsm_used)
    {
        Py_XDECREF(py_rsm);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beam_psf3(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"h","rt0","rs0","rtm","rsm","kj","local_dict", NULL};
    PyObject *py_h, *py_rt0, *py_rs0, *py_rtm, *py_rsm, *py_kj;
    int h_used, rt0_used, rs0_used, rtm_used, rsm_used, kj_used;
    py_h = py_rt0 = py_rs0 = py_rtm = py_rsm = py_kj = NULL;
    h_used= rt0_used= rs0_used= rtm_used= rsm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beam_psf3",const_cast<char**>(kwlist),&py_h, &py_rt0, &py_rs0, &py_rtm, &py_rsm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_rt0 = py_rt0;
        PyArrayObject* rt0_array = convert_to_numpy(py_rt0,"rt0");
        conversion_numpy_check_type(rt0_array,PyArray_DOUBLE,"rt0");
        conversion_numpy_check_size(rt0_array,1,"rt0");
        blitz::Array<double,1> rt0 = convert_to_blitz<double,1>(rt0_array,"rt0");
        blitz::TinyVector<int,1> Nrt0 = rt0.shape();
        rt0_used = 1;
        py_rs0 = py_rs0;
        PyArrayObject* rs0_array = convert_to_numpy(py_rs0,"rs0");
        conversion_numpy_check_type(rs0_array,PyArray_DOUBLE,"rs0");
        conversion_numpy_check_size(rs0_array,1,"rs0");
        blitz::Array<double,1> rs0 = convert_to_blitz<double,1>(rs0_array,"rs0");
        blitz::TinyVector<int,1> Nrs0 = rs0.shape();
        rs0_used = 1;
        py_rtm = py_rtm;
        PyArrayObject* rtm_array = convert_to_numpy(py_rtm,"rtm");
        conversion_numpy_check_type(rtm_array,PyArray_DOUBLE,"rtm");
        conversion_numpy_check_size(rtm_array,2,"rtm");
        blitz::Array<double,2> rtm = convert_to_blitz<double,2>(rtm_array,"rtm");
        blitz::TinyVector<int,2> Nrtm = rtm.shape();
        rtm_used = 1;
        py_rsm = py_rsm;
        PyArrayObject* rsm_array = convert_to_numpy(py_rsm,"rsm");
        conversion_numpy_check_type(rsm_array,PyArray_DOUBLE,"rsm");
        conversion_numpy_check_size(rsm_array,2,"rsm");
        blitz::Array<double,2> rsm = convert_to_blitz<double,2>(rsm_array,"rsm");
        blitz::TinyVector<int,2> Nrsm = rsm.shape();
        rsm_used = 1;
        py_kj = py_kj;
        std::complex<double> kj = convert_to_complex(py_kj,"kj");
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<float> term2;
            int numpoints_grid = Nrtm[0];
            int numpoints = Nrsm[0];
            int nc = Nrtm[1];    
            float term1, expon, kj_freq;
            float r0, rsi, rti;
        
            kj_freq = kj.imag();
        
            for (int t=0; t<numpoints_grid; ++t) {
                for (int s=0; s<numpoints; ++s) {
                    term1 = 0;
                    term2 = 0;
            
                    for (int i=0; i<nc; ++i) {
                        rsi = rsm(s,i);
                        rti = rtm(t,i);
                        
                        term1 += 1/(rti*rti);
                       
                        expon = kj_freq * (rti-rsi);
                        term2 += (std::complex<float>(cosf(expon),sinf(expon))) / (rsi*rti);
                    }
                    r0 = rs0(s)/rt0(t);
                    h(t,s) = r0*r0 / (term1*term1) * (term2*conj(term2)).real();
                }
            }
        
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(rt0_used)
    {
        Py_XDECREF(py_rt0);
    }
    if(rs0_used)
    {
        Py_XDECREF(py_rs0);
    }
    if(rtm_used)
    {
        Py_XDECREF(py_rtm);
    }
    if(rsm_used)
    {
        Py_XDECREF(py_rsm);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* r_beam_psf4(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"h","rt0","rs0","rtm","rsm","kj","local_dict", NULL};
    PyObject *py_h, *py_rt0, *py_rs0, *py_rtm, *py_rsm, *py_kj;
    int h_used, rt0_used, rs0_used, rtm_used, rsm_used, kj_used;
    py_h = py_rt0 = py_rs0 = py_rtm = py_rsm = py_kj = NULL;
    h_used= rt0_used= rs0_used= rtm_used= rsm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:r_beam_psf4",const_cast<char**>(kwlist),&py_h, &py_rt0, &py_rs0, &py_rtm, &py_rsm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_DOUBLE,"h");
        conversion_numpy_check_size(h_array,2,"h");
        blitz::Array<double,2> h = convert_to_blitz<double,2>(h_array,"h");
        blitz::TinyVector<int,2> Nh = h.shape();
        h_used = 1;
        py_rt0 = py_rt0;
        PyArrayObject* rt0_array = convert_to_numpy(py_rt0,"rt0");
        conversion_numpy_check_type(rt0_array,PyArray_DOUBLE,"rt0");
        conversion_numpy_check_size(rt0_array,1,"rt0");
        blitz::Array<double,1> rt0 = convert_to_blitz<double,1>(rt0_array,"rt0");
        blitz::TinyVector<int,1> Nrt0 = rt0.shape();
        rt0_used = 1;
        py_rs0 = py_rs0;
        PyArrayObject* rs0_array = convert_to_numpy(py_rs0,"rs0");
        conversion_numpy_check_type(rs0_array,PyArray_DOUBLE,"rs0");
        conversion_numpy_check_size(rs0_array,1,"rs0");
        blitz::Array<double,1> rs0 = convert_to_blitz<double,1>(rs0_array,"rs0");
        blitz::TinyVector<int,1> Nrs0 = rs0.shape();
        rs0_used = 1;
        py_rtm = py_rtm;
        PyArrayObject* rtm_array = convert_to_numpy(py_rtm,"rtm");
        conversion_numpy_check_type(rtm_array,PyArray_DOUBLE,"rtm");
        conversion_numpy_check_size(rtm_array,2,"rtm");
        blitz::Array<double,2> rtm = convert_to_blitz<double,2>(rtm_array,"rtm");
        blitz::TinyVector<int,2> Nrtm = rtm.shape();
        rtm_used = 1;
        py_rsm = py_rsm;
        PyArrayObject* rsm_array = convert_to_numpy(py_rsm,"rsm");
        conversion_numpy_check_type(rsm_array,PyArray_DOUBLE,"rsm");
        conversion_numpy_check_size(rsm_array,2,"rsm");
        blitz::Array<double,2> rsm = convert_to_blitz<double,2>(rsm_array,"rsm");
        blitz::TinyVector<int,2> Nrsm = rsm.shape();
        rsm_used = 1;
        py_kj = py_kj;
        std::complex<double> kj = convert_to_complex(py_kj,"kj");
        kj_used = 1;
        /*<function call here>*/     
        
            std::complex<float> term2;
            int numpoints_grid = Nrtm[0];
            int numpoints = Nrsm[0];
            int nc = Nrtm[1];    
            float term1, expon, kj_freq;
            float r0, rsi, rti;
            
            kj_freq = kj.imag();
        
            for (int t=0; t<numpoints_grid; ++t) {
                for (int s=0; s<numpoints; ++s) {
                    term1 = 0;
                    term2 = 0;
            
                    for (int i=0; i<nc; ++i) {
                        rsi = rsm(s,i);
                        rti = rtm(t,i);
                        
                        term1 += 1/(rti*rti);
                       
                        expon = kj_freq * (rti-rsi);
                        term2 += (std::complex<float>(cosf(expon),sinf(expon))) / (rsi*rti);
                    }
                    r0 = rs0(s);
                    h(t,s) = r0*r0 / (nc*term1*term1) * (term2*conj(term2)).real();
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(rt0_used)
    {
        Py_XDECREF(py_rt0);
    }
    if(rs0_used)
    {
        Py_XDECREF(py_rs0);
    }
    if(rtm_used)
    {
        Py_XDECREF(py_rtm);
    }
    if(rsm_used)
    {
        Py_XDECREF(py_rsm);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* transfer(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"h","rt0","rtm","kj","local_dict", NULL};
    PyObject *py_h, *py_rt0, *py_rtm, *py_kj;
    int h_used, rt0_used, rtm_used, kj_used;
    py_h = py_rt0 = py_rtm = py_kj = NULL;
    h_used= rt0_used= rtm_used= kj_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOO|O:transfer",const_cast<char**>(kwlist),&py_h, &py_rt0, &py_rtm, &py_kj, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_h = py_h;
        PyArrayObject* h_array = convert_to_numpy(py_h,"h");
        conversion_numpy_check_type(h_array,PyArray_CDOUBLE,"h");
        conversion_numpy_check_size(h_array,3,"h");
        blitz::Array<std::complex<double> ,3> h = convert_to_blitz<std::complex<double> ,3>(h_array,"h");
        blitz::TinyVector<int,3> Nh = h.shape();
        h_used = 1;
        py_rt0 = py_rt0;
        PyArrayObject* rt0_array = convert_to_numpy(py_rt0,"rt0");
        conversion_numpy_check_type(rt0_array,PyArray_DOUBLE,"rt0");
        conversion_numpy_check_size(rt0_array,1,"rt0");
        blitz::Array<double,1> rt0 = convert_to_blitz<double,1>(rt0_array,"rt0");
        blitz::TinyVector<int,1> Nrt0 = rt0.shape();
        rt0_used = 1;
        py_rtm = py_rtm;
        PyArrayObject* rtm_array = convert_to_numpy(py_rtm,"rtm");
        conversion_numpy_check_type(rtm_array,PyArray_DOUBLE,"rtm");
        conversion_numpy_check_size(rtm_array,2,"rtm");
        blitz::Array<double,2> rtm = convert_to_blitz<double,2>(rtm_array,"rtm");
        blitz::TinyVector<int,2> Nrtm = rtm.shape();
        rtm_used = 1;
        py_kj = py_kj;
        PyArrayObject* kj_array = convert_to_numpy(py_kj,"kj");
        conversion_numpy_check_type(kj_array,PyArray_CDOUBLE,"kj");
        conversion_numpy_check_size(kj_array,1,"kj");
        blitz::Array<std::complex<double> ,1> kj = convert_to_blitz<std::complex<double> ,1>(kj_array,"kj");
        blitz::TinyVector<int,1> Nkj = kj.shape();
        kj_used = 1;
        /*<function call here>*/     
        
            int numpoints = Nrtm[0];
            int nc = Nrtm[1];    
            int numfreq = Nkj[0];
            float expon, kj_freq, r0, factor, ri;
            
            for (int i_freq=0; i_freq<numfreq; ++i_freq) {
                kj_freq = (kj(i_freq)).imag();
                
                for (int t=0; t<numpoints; ++t) {
                    r0 = rt0(t);
                    
                    for (int i=0; i<nc; ++i) {
                        ri = rtm(t,i);
                        factor = r0/ri;
                        expon = kj_freq * ( r0 - ri );                 
                        h(i_freq,t,i) = factor*(std::complex<float>(cosf(expon),sinf(expon)));
                        
                    }
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(h_used)
    {
        Py_XDECREF(py_h);
    }
    if(rt0_used)
    {
        Py_XDECREF(py_rt0);
    }
    if(rtm_used)
    {
        Py_XDECREF(py_rtm);
    }
    if(kj_used)
    {
        Py_XDECREF(py_kj);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* gseidel(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"A","y","x","n","om","local_dict", NULL};
    PyObject *py_A, *py_y, *py_x, *py_n, *py_om;
    int A_used, y_used, x_used, n_used, om_used;
    py_A = py_y = py_x = py_n = py_om = NULL;
    A_used= y_used= x_used= n_used= om_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOO|O:gseidel",const_cast<char**>(kwlist),&py_A, &py_y, &py_x, &py_n, &py_om, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_A = py_A;
        PyArrayObject* A_array = convert_to_numpy(py_A,"A");
        conversion_numpy_check_type(A_array,PyArray_DOUBLE,"A");
        conversion_numpy_check_size(A_array,2,"A");
        blitz::Array<double,2> A = convert_to_blitz<double,2>(A_array,"A");
        blitz::TinyVector<int,2> NA = A.shape();
        A_used = 1;
        py_y = py_y;
        PyArrayObject* y_array = convert_to_numpy(py_y,"y");
        conversion_numpy_check_type(y_array,PyArray_DOUBLE,"y");
        conversion_numpy_check_size(y_array,1,"y");
        blitz::Array<double,1> y = convert_to_blitz<double,1>(y_array,"y");
        blitz::TinyVector<int,1> Ny = y.shape();
        y_used = 1;
        py_x = py_x;
        PyArrayObject* x_array = convert_to_numpy(py_x,"x");
        conversion_numpy_check_type(x_array,PyArray_DOUBLE,"x");
        conversion_numpy_check_size(x_array,1,"x");
        blitz::Array<double,1> x = convert_to_blitz<double,1>(x_array,"x");
        blitz::TinyVector<int,1> Nx = x.shape();
        x_used = 1;
        py_n = py_n;
        int n = convert_to_int(py_n,"n");
        n_used = 1;
        py_om = py_om;
        double om = convert_to_float(py_om,"om");
        om_used = 1;
        /*<function call here>*/     
        
            int numpoints=Ny[0];
            double x0;
            for (int i=0; i<n; ++i) {
                for (int j=0; j<numpoints; ++j) {
                    x0=0;
                    for (int k=0; k<j; ++k) {
                        x0+=A(j,k)*x(k);
                    };
                    for (int k=j+1; k<numpoints; ++k) {
                        x0+=A(j,k)*x(k);
                    };
                    x0=(1-om)*x(j)+om*(y(j)-x0);
                    x(j)=x0>0.0 ? x0 : 0;
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(A_used)
    {
        Py_XDECREF(py_A);
    }
    if(y_used)
    {
        Py_XDECREF(py_y);
    }
    if(x_used)
    {
        Py_XDECREF(py_x);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* gseidel1(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"A","y","x","n","local_dict", NULL};
    PyObject *py_A, *py_y, *py_x, *py_n;
    int A_used, y_used, x_used, n_used;
    py_A = py_y = py_x = py_n = NULL;
    A_used= y_used= x_used= n_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOO|O:gseidel1",const_cast<char**>(kwlist),&py_A, &py_y, &py_x, &py_n, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_A = py_A;
        PyArrayObject* A_array = convert_to_numpy(py_A,"A");
        conversion_numpy_check_type(A_array,PyArray_FLOAT,"A");
        conversion_numpy_check_size(A_array,2,"A");
        blitz::Array<float,2> A = convert_to_blitz<float,2>(A_array,"A");
        blitz::TinyVector<int,2> NA = A.shape();
        A_used = 1;
        py_y = py_y;
        PyArrayObject* y_array = convert_to_numpy(py_y,"y");
        conversion_numpy_check_type(y_array,PyArray_FLOAT,"y");
        conversion_numpy_check_size(y_array,1,"y");
        blitz::Array<float,1> y = convert_to_blitz<float,1>(y_array,"y");
        blitz::TinyVector<int,1> Ny = y.shape();
        y_used = 1;
        py_x = py_x;
        PyArrayObject* x_array = convert_to_numpy(py_x,"x");
        conversion_numpy_check_type(x_array,PyArray_FLOAT,"x");
        conversion_numpy_check_size(x_array,1,"x");
        blitz::Array<float,1> x = convert_to_blitz<float,1>(x_array,"x");
        blitz::TinyVector<int,1> Nx = x.shape();
        x_used = 1;
        py_n = py_n;
        int n = convert_to_int(py_n,"n");
        n_used = 1;
        /*<function call here>*/     
        
            int numpoints=Ny[0];
            float x0;
            for (int i=0; i<n; ++i) {
                for (int j=0; j<numpoints; ++j) {
                    x0=0;
                    for (int k=0; k<j; ++k) {
                        x0+=A(j,k)*x(k);
                    };
                    for (int k=j+1; k<numpoints; ++k) {
                        x0+=A(j,k)*x(k);
                    };
                    x0=(y(j)-x0);
                    x(j)=x0>0.0 ? x0 : 0;
                } 
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(A_used)
    {
        Py_XDECREF(py_A);
    }
    if(y_used)
    {
        Py_XDECREF(py_y);
    }
    if(x_used)
    {
        Py_XDECREF(py_x);
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                


static PyMethodDef compiled_methods[] = 
{
    {"faverage",(PyCFunction)faverage , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag",(PyCFunction)r_beamdiag , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull",(PyCFunction)r_beamfull , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_3d",(PyCFunction)r_beamdiag_3d , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_3d",(PyCFunction)r_beamfull_3d , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_classic",(PyCFunction)r_beamdiag_classic , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_classic",(PyCFunction)r_beamfull_classic , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_inverse",(PyCFunction)r_beamdiag_inverse , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_inverse",(PyCFunction)r_beamfull_inverse , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_os",(PyCFunction)r_beamdiag_os , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_os",(PyCFunction)r_beamfull_os , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_os_3d",(PyCFunction)r_beamdiag_os_3d , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_os_3d",(PyCFunction)r_beamfull_os_3d , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_os_classic",(PyCFunction)r_beamdiag_os_classic , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_os_classic",(PyCFunction)r_beamfull_os_classic , METH_VARARGS|METH_KEYWORDS},
    {"r_beamdiag_os_inverse",(PyCFunction)r_beamdiag_os_inverse , METH_VARARGS|METH_KEYWORDS},
    {"r_beamfull_os_inverse",(PyCFunction)r_beamfull_os_inverse , METH_VARARGS|METH_KEYWORDS},
    {"r_beam_psf",(PyCFunction)r_beam_psf , METH_VARARGS|METH_KEYWORDS},
    {"r_beam_psf1",(PyCFunction)r_beam_psf1 , METH_VARARGS|METH_KEYWORDS},
    {"r_beam_psf2",(PyCFunction)r_beam_psf2 , METH_VARARGS|METH_KEYWORDS},
    {"r_beam_psf3",(PyCFunction)r_beam_psf3 , METH_VARARGS|METH_KEYWORDS},
    {"r_beam_psf4",(PyCFunction)r_beam_psf4 , METH_VARARGS|METH_KEYWORDS},
    {"transfer",(PyCFunction)transfer , METH_VARARGS|METH_KEYWORDS},
    {"gseidel",(PyCFunction)gseidel , METH_VARARGS|METH_KEYWORDS},
    {"gseidel1",(PyCFunction)gseidel1 , METH_VARARGS|METH_KEYWORDS},
    {NULL,      NULL}        /* Sentinel */
};

PyMODINIT_FUNC initbeamformer(void)
{
    
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
    (void) Py_InitModule("beamformer", compiled_methods);
}

#ifdef __CPLUSCPLUS__
}
#endif
