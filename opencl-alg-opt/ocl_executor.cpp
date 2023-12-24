#include "ocl_executor.h"

std::string get_error_string(cl_int err){
     switch(err){
         case 0: return "CL_SUCCESS";
         case -1: return "CL_DEVICE_NOT_FOUND";
         case -2: return "CL_DEVICE_NOT_AVAILABLE";
         case -3: return "CL_COMPILER_NOT_AVAILABLE";
         case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
         case -5: return "CL_OUT_OF_RESOURCES";
         case -6: return "CL_OUT_OF_HOST_MEMORY";
         case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
         case -8: return "CL_MEM_COPY_OVERLAP";
         case -9: return "CL_IMAGE_FORMAT_MISMATCH";
         case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
         case -11: return "CL_BUILD_PROGRAM_FAILURE";
         case -12: return "CL_MAP_FAILURE";

         case -30: return "CL_INVALID_VALUE";
         case -31: return "CL_INVALID_DEVICE_TYPE";
         case -32: return "CL_INVALID_PLATFORM";
         case -33: return "CL_INVALID_DEVICE";
         case -34: return "CL_INVALID_CONTEXT";
         case -35: return "CL_INVALID_QUEUE_PROPERTIES";
         case -36: return "CL_INVALID_COMMAND_QUEUE";
         case -37: return "CL_INVALID_HOST_PTR";
         case -38: return "CL_INVALID_MEM_OBJECT";
         case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
         case -40: return "CL_INVALID_IMAGE_SIZE";
         case -41: return "CL_INVALID_SAMPLER";
         case -42: return "CL_INVALID_BINARY";
         case -43: return "CL_INVALID_BUILD_OPTIONS";
         case -44: return "CL_INVALID_PROGRAM";
         case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
         case -46: return "CL_INVALID_KERNEL_NAME";
         case -47: return "CL_INVALID_KERNEL_DEFINITION";
         case -48: return "CL_INVALID_KERNEL";
         case -49: return "CL_INVALID_ARG_INDEX";
         case -50: return "CL_INVALID_ARG_VALUE";
         case -51: return "CL_INVALID_ARG_SIZE";
         case -52: return "CL_INVALID_KERNEL_ARGS";
         case -53: return "CL_INVALID_WORK_DIMENSION";
         case -54: return "CL_INVALID_WORK_GROUP_SIZE";
         case -55: return "CL_INVALID_WORK_ITEM_SIZE";
         case -56: return "CL_INVALID_GLOBAL_OFFSET";
         case -57: return "CL_INVALID_EVENT_WAIT_LIST";
         case -58: return "CL_INVALID_EVENT";
         case -59: return "CL_INVALID_OPERATION";
         case -60: return "CL_INVALID_GL_OBJECT";
         case -61: return "CL_INVALID_BUFFER_SIZE";
         case -62: return "CL_INVALID_MIP_LEVEL";
         case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
         default: return "Unknown OpenCL error";
     }
 }
void CheckErrorAt(cl_int err,const char * source_info){
    if (err){
        std::cout << "Error: at " << source_info << ":\n" << get_error_string(err) << std::endl;
        exit(err);
    }
}
cl_int CLEnqueueBarrier(cl_command_queue command_queue){
#if CL_TARGET_OPENCL_VERSION >= 120
    return clEnqueueBarrierWithWaitList(command_queue,0,NULL,NULL);
#else
    return clEnqueueBarrier(command_queue);
#endif
}
cl_command_queue CLCreateCommandQueue(cl_context context, cl_device_id device, cl_int * errcode_ret){
#if CL_TARGET_OPENCL_VERSION >= 200
    return clCreateCommandQueueWithProperties(context,device,NULL,errcode_ret);
#else
    return clCreateCommandQueue(context,device,0,errcode_ret);
#endif
}

CLBuffer::CLBuffer(cl_context context,cl_command_queue queue,size_t size, size_t element_size):
    bufsize(size),
    mycontext(context),
    myqueue(queue),
    el_size(element_size)
    {

    cl_int error = CL_SUCCESS;
    buf = clCreateBuffer(context,CL_MEM_READ_WRITE,bytes(),nullptr,&error);
    CheckError(error);
}

void CLBuffer::copy_buffer(CLBuffer src_buf){
    assert(src_buf.bufsize == this->bufsize);
    assert(src_buf.myqueue == this->myqueue);
    assert(src_buf.mycontext == this->mycontext);
    assert(src_buf.el_size == this->el_size);
    CheckError(CLEnqueueBarrier(myqueue));
    CheckError(clEnqueueCopyBuffer(myqueue,
                        src_buf.buf,this->buf,
                        0,0,
                        bytes(),
                        0,nullptr,
                        nullptr));
}

CL_NDRange::CL_NDRange(size_t in_x,size_t in_y, size_t in_z){
    x = in_x;
    y = in_y;
    z = in_z;
    ndim = 3;
    assert(in_x != 0);
    assert(in_y != 0);
    assert(in_z != 0);
}
CL_NDRange::CL_NDRange(size_t in_x,size_t in_y){
    x = in_x;
    y = in_y;
    z = -1;
    ndim = 2;
    assert(in_x != 0);
    assert(in_y != 0);
}
CL_NDRange::CL_NDRange(size_t in_x){
    x = in_x;
    y = -1;
    z = -1;
    ndim = 1;
    assert(in_x != 0);
}
CL_NDRange::CL_NDRange(){
    x = -1;
    y = -1;
    z = -1;
    ndim = 0;
}
size_t * CL_NDRange::array_view(){
    assert(ndim != 0);
    return reinterpret_cast<size_t*>(this);
}
cl_uint CL_NDRange::dim(){
    return ndim;
}

CLKernel::CLKernel(cl_program in_prog,cl_command_queue in_queue,const char * kern_name,CL_NDRange in_run_range,CL_NDRange in_work_group_range,std::vector<cl_mem> args){
    myqueue = in_queue;
    program = in_prog;
    run_range = in_run_range;
    work_group_range = in_work_group_range;

    assert(run_range.dim() > 0 && "run_range needs to have at least 1 dimention specified");

    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateKernel.html
    cl_int error = CL_SUCCESS;
    kern = clCreateKernel (program, kern_name, &error);
    CheckError (error);

    int idx = 0;
    using namespace std;
    for(cl_mem b_info: args){
        CheckError(clSetKernelArg(kern,idx,sizeof(b_info),&b_info));
        idx++;
    }
}
void CLKernel::run(){
    CheckError(CLEnqueueBarrier(myqueue));
    size_t * work_group_ptr = work_group_range.dim() == 0 ? NULL : work_group_range.array_view();
    CheckError(clEnqueueNDRangeKernel(
                   myqueue,
                   kern,
                   run_range.dim(),
                   nullptr,
                   run_range.array_view(),
                   work_group_ptr,
                   0,nullptr,
                   nullptr
                   ));
}

OpenCLExecutor::OpenCLExecutor(std::string in_source_path, std::string in_compile_ops)
{
    source_path = in_source_path;
    compile_ops = in_compile_ops;
    build_program();
    std::cout << "finished building program" << std::endl;
}
OpenCLExecutor::~OpenCLExecutor(){
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
}
void OpenCLExecutor::get_main_device(){
    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);

    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platform found" << std::endl;
        exit(1);
    } else {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

    for (cl_uint i = 0; i < platformIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
    }
    cl_platform_id platformId = platformIds [0];

    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformId, CL_DEVICE_TYPE_ALL, 0, nullptr,
        &deviceIdCount);

    if (deviceIdCount == 0) {
        std::cerr << "No OpenCL devices found" << std::endl;
        exit(1);
    } else {
        std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
    }

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformId, CL_DEVICE_TYPE_ALL, deviceIdCount,
        deviceIds.data (), nullptr);

    for (cl_uint i = 0; i < deviceIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
    }
    cl_device_id deviceId = deviceIds [0];

    std::cout << "Using platform: "<< GetPlatformName (platformId)<<"\n";
    std::cout << "Using device: "<< GetDeviceName (deviceId)<<"\n";

    this->platform = platformId;
    this->device = deviceId;
}
void OpenCLExecutor::create_context(){
    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (this->platform),
        0, 0
    };

    cl_int error = CL_SUCCESS;
    this->context = clCreateContext (contextProperties, 1,
        &device, nullptr, nullptr, &error);
    CheckError(error);
    std::cout << "Context created" << std::endl;
}
void OpenCLExecutor::create_queue(){
    // https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clCreateCommandQueueWithProperties.html

    cl_int error = CL_SUCCESS;
    this->queue = CLCreateCommandQueue (context, this->device, &error);
    CheckError (error);
}
void OpenCLExecutor::build_program(){
    get_main_device();
    create_context();
    create_queue();
    CreateProgram();
}

std::string OpenCLExecutor::get_source(){
    std::ifstream file(source_path);
    if(!file){
        std::cout << "the file " << source_path << " is missing!\n";
        exit(1);
    }
    //slow way to read a file (but file size is small)
    std::string fstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    file.close();
    return fstr;
}

void OpenCLExecutor::CreateProgram ()
{
    std::string source = get_source();
    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateProgramWithSource.html
    size_t lengths [1] = { source.size () };
    const char* sources [1] = { source.data () };

    cl_int error = CL_SUCCESS;
    this->program = clCreateProgramWithSource (this->context, 1, sources, lengths, &error);
    CheckError (error);

    cl_int build_error = clBuildProgram (program, 1, &this->device,
        this->compile_ops.c_str(), nullptr, nullptr);

    if(build_error == CL_BUILD_PROGRAM_FAILURE){
        size_t len = 0;
        cl_int ret = CL_SUCCESS;
        CheckError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
        std::vector<char> data(len);
        CheckError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, data.data(), NULL));
        std::cout << "Build error:\n" << std::string(data.begin(),data.end()) << std::endl;
        exit(1);
    }
    else{
        CheckError(build_error);
    }
    //ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
}

 std::string OpenCLExecutor::GetPlatformName (cl_platform_id id)
 {
     size_t size = 0;
     clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

     std::string result;
     result.resize (size);
     clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
         const_cast<char*> (result.data ()), nullptr);

     return result;
 }

 std::string OpenCLExecutor::GetDeviceName (cl_device_id id)
 {
     size_t size = 0;
     clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

     std::string result;
     result.resize (size);
     clGetDeviceInfo (id, CL_DEVICE_NAME, size,
         const_cast<char*> (result.data ()), nullptr);

     return result;
 }
