# CNN
#cnn_opencl.c
#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cnn.h"

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    }

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */

cl_uint platformCount;
cl_platform_id* platforms;
cl_uint deviceCount;
cl_device_id* devices;
cl_device_id device;
cl_context context;
cl_int err;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_fc;
cl_mem input_buf, output_buf;

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file;

	file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(1);
	}

	fseek(file, 0, SEEK_END);

	length = (size_t)ftell(file);

	rewind(file);
	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}

	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

static void pooling2x2(float* input, float* output, int N) {

	///*
	//step 2. 메모리 초기화. CL 커널 버퍼 초기화
	//*/
	//int n_input_size = N * N * 4;
	//int n_output_size = N * N;

	//input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n_input_size, input, &err);
	//output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n_output_size, NULL, &err);

	///*
	//	step 3. 커널 인자 mapping
	//*/
	//err = clSetKernelArg(kernel_pooling, 0, sizeof(cl_mem), &input_buf);
	//CHECK_ERROR(err);
	//err = clSetKernelArg(kernel_pooling, 1, sizeof(cl_mem), &output_buf);
	//CHECK_ERROR(err);
	//err = clSetKernelArg(kernel_pooling, 2, sizeof(int), &N);
	//CHECK_ERROR(err);

	///*
	//	step 4. 커맨드 큐 삽입
	//*/
	//size_t global_size[2] = { N, N };
	//size_t local_size[2] = { 1, 1 };
	//err = clEnqueueNDRangeKernel(queue, kernel_pooling, 2, NULL, global_size, NULL, 0, NULL, NULL);
	//CHECK_ERROR(err);

	///*
	//	step 5. 커널 버퍼 읽기
	//*/
	//err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * n_output_size, output, 0, NULL, NULL);
	//CHECK_ERROR(err);
	int i, j, k, l;
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				float max = 0;
				for (k = 0; k < 2; k++) {
					for (l = 0; l < 2; l++) {
						float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
						max = (max > pixel) ? max : pixel;
					}
				}
				output[i * N + j] = max;
			}
		}
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float* inputs, float* outputs, int D, int N) {

	int i;
	for (i = 0; i < D; i++) {
		float* input = inputs + i * N * N * 4;
		float* output = outputs + i * N * N;
		pooling2x2(input, output, N);
	}
}

static void convolution3x3(float* input, float* output, float* filter, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float sum = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
						sum += input[x * N + y] * filter[k * 3 + l];
				}
			}
			output[i * N + j] += sum;
		}
	}
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
	int i, j;

	memset(outputs, 0, sizeof(float) * N * N * D2);

	for (j = 0; j < D2; j++) {
		for (i = 0; i < D1; i++) {
			float* input = inputs + N * N * i;
			float* output = outputs + N * N * j;
			float* filter = filters + 3 * 3 * (j * D1 + i);
			convolution3x3(input, output, filter, N);
		}
	}

	for (i = 0; i < D2; i++) {
		float* output = outputs + N * N * i;
		float bias = biases[i];
		for (j = 0; j < N * N; j++) {
			output[j] = ReLU(output[j] + bias);
		}
	}
}

/*
 * M = output size
 * N = input size
 */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
	
	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), input_neuron, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, M * sizeof(float), NULL, NULL);
	cl_mem weightBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, M * N * sizeof(float), weights, NULL);
	cl_mem biasesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, M * sizeof(float), biases, NULL);
	
	err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), &inputBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), &outputBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), &weightBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 3, sizeof(cl_mem), &biasesBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 4, sizeof(int), &N);
	CHECK_ERROR(err);

	size_t global_size = M;
	
	err = clEnqueueNDRangeKernel(queue, kernel_fc, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(float) * M, output_neuron, 0, NULL, NULL);
	CHECK_ERROR(err);

	clReleaseMemObject(outputBuffer);
	clReleaseMemObject(inputBuffer);
	clReleaseMemObject(weightBuffer);
	clReleaseMemObject(biasesBuffer);
}

static void softmax(float* output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}
void cnn_init() {
	/*
	 * TODO
	 * Initialize OpenCL objects as global variables. For example,
	 * clGetPlatformIDs(1, &platform, NULL);
	 */

	 // 1. platform 가져오기
	err = clGetPlatformIDs(0, NULL, &platformCount);
	CHECK_ERROR(err);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);

	err = clGetPlatformIDs(platformCount, platforms, NULL);
	CHECK_ERROR(err);

	// 2. device 가져오기
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	CHECK_ERROR(err);

	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	CHECK_ERROR(err);

	device = devices[0];

	// 3. context 생성하기
	context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	// command queue create!!
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	size_t kernel_source_size;

	/*
		step 1. 커널 생성과 CL파일 객체 가져오기
	*/
	char* kernel_source = get_source_code("fc.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
		&kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, NULL, NULL,  NULL);
	
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		//get log size from build info log first.
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
			NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		//get log from build info log.
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		//show the log
		printf("comfiler error : \n%s\n", log);
		free(log);
		exit(0);
	}
	CHECK_ERROR(err);
	kernel_fc = clCreateKernel(program, "fullConnectionKernel", &err);

}


void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.
	 * See "cnn_seq.c" if you don't know what to do.
	 */

	float* w1_1, * b1_1, * w1_2, * b1_2;
	float* w2_1, * b2_1, * w2_2, * b2_2;
	float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
	float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
	float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
	float* w1, * b1, * w2, * b2, * w3, * b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	// allocate memory for output of each layer
	float* c1_1, * c1_2, * p1;
	float* c2_1, * c2_2, * p2;
	float* c3_1, * c3_2, * c3_3, * p3;
	float* c4_1, * c4_2, * c4_3, * p4;
	float* c5_1, * c5_2, * c5_3, * p5;
	float* fc1, * fc2, * fc3;
	c1_1 = alloc_layer(64 * 32 * 32);
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);

	// run network
	for (int i = 0; i < num_images; ++i)
	{
		float* image = images + i * 3 * 32 * 32;

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
		pooling_layer(c1_2, p1, 64, 16);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10);

		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];
	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);

	clReleaseMemObject(input_buf);
	clReleaseMemObject(output_buf);
	clReleaseKernel(kernel_fc);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(platforms);
}

#softmax
void softmax(float* output, int n) {
	int i;
	float max = output[0];
	for (i = 1; i < n; i++) {
		max = (output[i] > max) ? output[i] : max;
	}

	const char* sourcefile = "soft_max.cl";
	const char* kernelname = "soft_max_kernel";

	cl_command_queue queue = clcreatecommandqueuewithproperties(context, device, 0, null);
	size_t kernel_source_size;

	char* kernel_source = get_source_code(sourcefile, &kernel_source_size);

	cl_program program = clcreateprogramwithsource(context, 1, (const char**)&kernel_source, &kernel_source_size, null);
	cl_int build_status = clbuildprogram(program, 1, &device, null, null, null);
	cl_kernel kernel = clcreatekernel(program, kernelname, &err);

	cl_mem outputbuffer = clcreatebuffer(context, cl_mem_read_write, n * sizeof(int), null, null);

	err = clsetkernelarg(kernel, 0, sizeof(cl_mem), &outputbuffer);
	err = clsetkernelarg(kernel, 1, sizeof(cl_float), &max);

	size_t global_size[1] = { n };

	err = clenqueuendrangekernel(queue, kernel, 1, null, global_size, 0, 0, null, null);
	err = clenqueuereadbuffer(queue, outputbuffer, cl_true, 0, sizeof(float) * n, output, 0, null, null);

	clflush(queue);
	clfinish(queue);

	system("pause");

	clreleasememobject(outputbuffer);
	clreleasekernel(kernel);
	clreleaseprogram(program);
	clreleasecommandqueue(queue);
	clreleasecontext(context);
}

#fc.cl
#define ReLU(x) (((x)>0)?(x):0)
__kernel void fullConnectionKernel(__global float* inputs, __global float* output,
	__global float* weights, __global float* biases, int N) {

	int j = get_global_id(0);
	
	float sum = 0;
	for(int i = 0; i < N; i++) {
		sum += inputs[i] * weights[j * N + i];
	}
    
	sum += biases[j];
	// output[j] = ((output[j]) > 0) ? (output[j]) : 0;
    // if (sum > 0) output[j] = sum; //ReLU
    // else output[j] = 0;
    output[j] = ReLU(sum);

}
