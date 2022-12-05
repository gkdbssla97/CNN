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