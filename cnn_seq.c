#include<cl/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cnn.h"

#define _crt_secure_no_warnings
#define relu(x) (((x)>0)?(x):0)

static void pooling2x2(float* input, float* output, int n) {
	int i, j, k, l;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float max = 0;
			for (k = 0; k < 2; k++) {
				for (l = 0; l < 2; l++) {
					float pixel = input[(i * 2 + k) * 2 * n + j * 2 + l];
					max = (max > pixel) ? max : pixel;
				}
			}
			output[i * n + j] = max;
		}
	}
}

/*
 * d = channel size
 * n = width and height of an output image
 * thus, input is (d, n * 2, n * 2) and output is (d, n, n).
 */
static void pooling_layer(float* inputs, float* outputs, int d, int n) {
	int i;
	for (i = 0; i < d; i++) {
		float* input = inputs + i * n * n * 4;
		float* output = outputs + i * n * n;
		pooling2x2(input, output, n);
	}
}

static void convolution3x3(float* input, float* output, float* filter, int n) {
	int i, j, k, l;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float sum = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < n && y >= 0 && y < n)
						sum += input[x * n + y] * filter[k * 3 + l];
				}
			}
			output[i * n + j] += sum;
		}
	}
}

/*
 * d2 = output channel size
 * d1 = input channel size
 * n = width and height of an input image
 * input image is zero-padded by 1.
 * thus, input is (d1, n, n) and output is (d2, n, n)
 */
#define relu(x) (((x)>0)?(x):0)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int d2, int d1, int n) {
	int i, j;

	memset(outputs, 0, sizeof(float) * n * n * d2);

	for (j = 0; j < d2; j++) {
		for (i = 0; i < d1; i++) {
			float* input = inputs + n * n * i;
			float* output = outputs + n * n * j;
			float* filter = filters + 3 * 3 * (j * d1 + i);
			convolution3x3(input, output, filter, n);
		}
	}

	for (i = 0; i < d2; i++) {
		float* output = outputs + n * n * i;
		float bias = biases[i];
		for (j = 0; j < n * n; j++) {
			output[j] = relu(output[j] + bias);
		}
	}
}

/*
 * m = output size
 * n = input size
 */

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	file* file = fopen(file_name, "r");
	if (file == null) {
		printf("[%s:%d] failed to open %s\n", __file__, __line__, file_name);
		exit(exit_failure);
	}
	fseek(file, 0, seek_end);
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

static int platformnum = 0;
static int devicenum = 0;


static cl_uint platformcount;
static cl_platform_id* platforms;
static cl_uint devicecount;
static cl_device_id* devices;
static cl_device_id device;
static cl_context context;
static cl_int err;

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	// 1. platform 가져오기
	clgetplatformids(0, null, &platformcount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformcount);
	clgetplatformids(platformcount, platforms, null);

	// 2. device 가져오기
	clgetdeviceids(platforms[0], cl_device_type_all, 0, null, &devicecount);
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * devicecount);
	clgetdeviceids(platforms[platformnum], cl_device_type_all, devicecount, devices, null);
	device = devices[devicenum];

	// 3. context 생성하기
	context = clcreatecontext(null, 1, &device, null, null, null);

}

void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int m, int n) {

	const char* sourcefile = "fc.cl";
	const char* kernelname = "fullconcectionkernel";

	cl_command_queue queue = clcreatecommandqueuewithproperties(context, device, 0, null);
	size_t kernel_source_size;

	char* kernel_source = get_source_code(sourcefile, &kernel_source_size);

	cl_program program = clcreateprogramwithsource(context, 1, (const char**)&kernel_source, &kernel_source_size, null);
	cl_int build_status = clbuildprogram(program, 1, &device, null, null, null);
	cl_kernel kernel = clcreatekernel(program, kernelname, &err);

	cl_mem inputbuffer = clcreatebuffer(context, cl_mem_read_write, n * sizeof(int), null, null);
	cl_mem outputbuffer = clcreatebuffer(context, cl_mem_read_write, n * sizeof(int), null, null);
	cl_mem weightbuffer = clcreatebuffer(context, cl_mem_read_write, n * sizeof(int), null, null);
	cl_mem biasesbuffer = clcreatebuffer(context, cl_mem_read_write, n * sizeof(int), null, null);

	err = clsetkernelarg(kernel, 0, sizeof(cl_mem), &outputbuffer);
	err = clsetkernelarg(kernel, 1, sizeof(cl_mem), &inputbuffer);
	err = clsetkernelarg(kernel, 2, sizeof(cl_mem), &weightbuffer);
	err = clsetkernelarg(kernel, 3, sizeof(cl_mem), &biasesbuffer);
	err = clsetkernelarg(kernel, 4, sizeof(cl_int), &n);

	size_t global_size[1] = { n };

	err = clenqueuendrangekernel(queue, kernel, 1, null, global_size, 0, 0, null, null);
	err = clenqueuereadbuffer(queue, outputbuffer, cl_true, 0, sizeof(float) * n, output_neuron, 0, null, null);

	clflush(queue);
	clfinish(queue);

	system("pause");

	clreleasememobject(outputbuffer);
	clreleasekernel(kernel);
	clreleaseprogram(program);
	clreleasecommandqueue(queue);
	clreleasecontext(context);
}

static int find_max(float* fc, int n) {
	
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < n; i++) {
		if (maxval < fc[i]) {
				maxval = fc[i];
				maxid = i;
			}
		}
		return maxid;
}

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

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	// slice the network into weights and biases
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
}
