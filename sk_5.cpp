// Compile on GPU with: nvcc -x cu -lcublasLt -arch=[RELEVANT COMPUTE ARCHITECTURE, EX:compute_89] -o sk_5 sk_5.cpp

#include <memory>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <random>
#include <chrono>

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

#include <mma.h>

using namespace nvcuda;

using __dtype = __half;

inline void checkCudaStatus(cudaError_t status) {
	if (status != cudaSuccess) {
		printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
		throw std::logic_error("cuda API failed");
	}
}

__device__
__half activ_dev(__half x) {
	return __half(tanh((float)x));
}

__half activ(__half x) {
	return __half(tanh((float)x));
}

__device__
__half pow_dev(__half x, __half y) { // x^y
	return hexp(y * hlog(x));
}

__device__
__half pow10_dev(__half y) { // 10^y
	// ln(10) ~= 2.30258509299404568
	return hexp(y * __half(2.302585));
}

__half hardtanh(__half x) {
	return x > __half(1.0) ? __half(1.0) : (x < __half(-1.0) ? __half(-1.0) : x);
}

__device__
__half hardtanh_dev(__half x) {
	return x > __half(1.0) ? __half(1.0) : (x < __half(-1.0) ? __half(-1.0) : x);
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8
constexpr int blockM = NUM_WARPS * WMMA_M; // 128

__constant__ __half W11_c[WMMA_K];
__constant__ __half W12_c[WMMA_K];
__constant__ __half W13_c[WMMA_K];

// __constant__ __half W3_c[WMMA_N];
__constant__ __half W4_c[WMMA_N];

__constant__ __half b1_c[WMMA_K];
__constant__ __half b2_c[WMMA_N];
__constant__ __half b3_c[WMMA_K];
__constant__ __half b4_c[1];


__global__ void wmmaKernel(half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, half *__restrict D,
						   const half *__restrict__ x1, const half *__restrict__ x2, const half *__restrict__ x3,
						   const half *__restrict__ W11, const half *__restrict__ W12, const half *__restrict__ W13,
						   const half *__restrict__ W3, const half *__restrict__ W4,
						   const half *__restrict__ b1, const half *__restrict__ b2,
						   const half *__restrict__ b3, const half *__restrict__ b4,
						   half *__restrict__ y, size_t M, size_t N, size_t K) {

	int warp_num = threadIdx.x / WARP_SIZE;
	int blockK = WMMA_K; 
	int offsetM = blockM * blockIdx.x;

	extern __shared__ __half shared_data[];
	__half* A_s = shared_data;
	__half* x1_s = shared_data + blockM * WMMA_K;
	__half* x2_s = shared_data + blockM * WMMA_K + blockM;
	__half* x3_s = shared_data + blockM * WMMA_K + 2 * blockM;

	if (threadIdx.x < blockM) {
		x1_s[threadIdx.x] = x1[threadIdx.x + offsetM];
		x2_s[threadIdx.x] = x2[threadIdx.x + offsetM];
		x3_s[threadIdx.x] = x3[threadIdx.x + offsetM];
	}

	__syncthreads();

	// A[M, K] = x[M, 2] * W1[2, K] + b1[1, K]

	if (threadIdx.x < blockM) {
		for (int j = 0; j < blockK; j++) {
			A_s[threadIdx.x * blockK + j] = activ_dev(x1_s[threadIdx.x] * W11_c[j] + x2_s[threadIdx.x] * W12_c[j] + x3_s[threadIdx.x] * W13_c[j] + b1_c[j]);
		} 
	}

	__syncthreads();

	// C[M, N] = A[M, K] * B[K, N]

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
	wmma::load_matrix_sync(b_frag, B, WMMA_K);
	wmma::fill_fragment(c_frag, 0.0f);

	wmma::load_matrix_sync(a_frag, A_s + warp_num * WMMA_M * WMMA_K, WMMA_K);
	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	wmma::store_matrix_sync(A_s + warp_num * WMMA_M * WMMA_K, c_frag, WMMA_K, wmma::mem_row_major);

	__syncthreads();

	// C[M, N] = activ(C[M, N] + b2[1, N])
	if (threadIdx.x < blockM) {
		for (int j = 0; j < N; j++) {
			A_s[threadIdx.x * blockK + j] = activ_dev(A_s[threadIdx.x * blockK + j] + b2_c[j]);
		}
	}

	__syncthreads();

	// D[M, N] = C[M, K] * W3[K, N]
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> d_frag;
	wmma::load_matrix_sync(b_frag, W3, WMMA_K);
	wmma::fill_fragment(d_frag, 0.0f);

	wmma::load_matrix_sync(a_frag, A_s + warp_num * WMMA_M * WMMA_K, WMMA_K);
	wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
	wmma::store_matrix_sync(A_s + warp_num * WMMA_M * WMMA_K, d_frag, WMMA_K, wmma::mem_row_major);

	__syncthreads();


	// y[M, 1] = hardtanh( activ(D[M, N] + b3[1, N]) * W4[N, 1] + b4[1, 1] )
	if (threadIdx.x < blockM) {
		half y_tmp1 = __half(0.0);
		for (int j = 0; j < N; j++) {
			y_tmp1 += activ_dev(A_s[threadIdx.x * blockK + j] + b3_c[j]) * W4_c[j];
		}
		y[threadIdx.x + offsetM] = hardtanh_dev(y_tmp1 + b4_c[0]);
	}

	__syncthreads();

}

void wmmaCall(half *A, half *W2, half *C, half *D,
			  half *x1, half *x2, half *x3,
			  half *W11, half *W12, half *W13, half *W3, half *W4,
			  half *b1, half *b2, half *b3, half *b4, half *y,
			  size_t M, size_t N, size_t K) {
	dim3 block(BLOCK_SIZE);
	dim3 grid((M / WMMA_M / NUM_WARPS));
	// Shared memory: (blockM * K) + (3 * blockM)
	wmmaKernel<<<(M / WMMA_M / NUM_WARPS), BLOCK_SIZE, 
				 blockM * WMMA_K * sizeof(half) + 
				 3 * blockM * sizeof(half)>>>(A, W2, C, D, x1, x2, x3, W11, W12, W13, W3, W4,
				 b1, b2, b3, b4, y, M, N, K);
	// printf("%d, %d\n", grid.x, block.x);
	checkCudaStatus(cudaGetLastError());
}

void compare(const half *A, const half *B, size_t size, half tol = 1e-3) {
	int mist = 0;
	for (int i = 0; i < size; i++) {
		if (__habs(A[i] - B[i]) > tol) {
			if(mist < 20){
				printf("Error at %d: %f, expect %f\n", i, float(A[i]), float(B[i]));
			}
			mist++;
		}
	}
	if (mist == 0) {
		printf("No error\n");
		// for(int i = 0; i < 10; i++){
		// 	std::cout << i<< ": " << float(A[i]) << " = " << float(B[i]) << std::endl;
		// }
	} else {
		printf("%d errors\n", mist);
	}
}

int main(int argc, char* argv[]) {

	if(argc < 4){
		std::cout << "Usage: ./sk_5 <log gamma> <log epsilon> <outputFileName>" << std::endl;
		return 1;
	}
	float G = std::stof(argv[1]);
	float E = std::stof(argv[2]);
	std::string outputFileName = argv[3];

	// A[M, K] * B[K, N] = C[M, N]

	constexpr size_t M = 1ul<<24;
	// constexpr size_t M = 1024;

	// int write_output = 1;
	int write_output = 0;
	// int cpu_comparison = 1;
	int cpu_comparison = 0;
	constexpr int K = WMMA_K;
	constexpr int N = WMMA_N;
	printf("M: %lu, M*N: %lu\n", M, M*K); 

	std::vector<__dtype> W11_h(K);
	std::vector<__dtype> W12_h(K);
	std::vector<__dtype> W13_h(K);
	std::vector<__dtype> A_h(M * K);
	std::vector<__dtype> W2_h(K * N);
	std::vector<__dtype> W3_h(K * N);
	std::vector<__dtype> C_h(M * N);
	std::vector<__dtype> D_h(M * N);
	std::vector<__dtype> W4_h(N);
	std::vector<__dtype> b1_h(K);
	std::vector<__dtype> b2_h(K);
	std::vector<__dtype> b3_h(K);
	std::vector<__dtype> b4_h(1);

	std::vector<__dtype> x1_h(M);
	std::vector<__dtype> x2_h(M);
	std::vector<__dtype> x3_h(M);
	std::vector<__dtype> y_h(M);
	std::vector<__dtype> y_out_h(M);

	std::string filename = "net1_layer0.txt";
	std::ifstream file(filename);
	std::string line;
	std::getline(file, line);
	std::vector<std::string> tokens;
	std::string token;
	std::stringstream token_stream(line);
	while (std::getline(token_stream, token, ' ')) {
		tokens.push_back(token);
	}
	// Set values in W11
	for (int i = 0; i < 16; i++) {
		W11_h[i] = __half(std::stof(tokens[i]));
	}
	// Set values in W12
	for (int i = 0; i < 16; i++) {
		W12_h[i] = __half(std::stof(tokens[i + 16]));
	}
	// Set values in W13
	for (int i = 0; i < 16; i++) {
		W13_h[i] = __half(std::stof(tokens[i + 32]));
	}
	// Set values in b1
	for (int i = 0; i < 16; i++){
		std::string line;
		std::getline(file, line);
		b1_h[i] = __half(std::stof(line));
	}
	tokens.clear();
	file.close();
	
	filename = "net1_layer1.txt";
	std::ifstream file2(filename);
	std::getline(file2, line);
	std::stringstream token_stream2(line);
	while (std::getline(token_stream2, token, ' ')) {
		tokens.push_back(token);
	}
	// Set values in B / W2
	int count = 0;
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++){
			W2_h[i * 16 + j] = __half(std::stof(tokens[i + 16 * j]));
			count++;
		}
	}
	std::cout << "Set " << count << " values in W2" << std::endl;
	// Set values in b2
	count=0;
	for (int i = 0; i < 16; i++){
		std::string line;
		std::getline(file2, line);
		b2_h[i] = __half(std::stof(line));
		// std::cout << float(b2_h[i]) << " ";
		count++;
	}
	std::cout << "Set " << count << " values in b2" << std::endl;
	tokens.clear();
	file2.close();

	filename = "net1_layer2.txt";
	std::ifstream file3(filename);
	std::getline(file3, line);
	std::stringstream token_stream3(line);
	while (std::getline(token_stream3, token, ' ')) {
		tokens.push_back(token);
	}
	// Set values in W3
	count = 0;
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++){
			W3_h[i * 16 + j] = __half(std::stof(tokens[i + 16 * j]));
			count++;
		}
	}
	std::cout << "Set " << count << " values in W3" << std::endl;
	count=0;
	for (int i = 0; i < 16; i++){
		std::string line;
		std::getline(file3, line);
		b3_h[i] = __half(std::stof(line));
		count++;
	}
	std::cout << "Set " << count << " values in b3" << std::endl;
	tokens.clear();
	file3.close();

	filename = "net1_layer3.txt";
	std::ifstream file4(filename);
	std::getline(file4, line);
	std::stringstream token_stream4(line);
	while (std::getline(token_stream4, token, ' ')) {
		tokens.push_back(token);
	}
	// Set values in W4
	count = 0;
	for (int i = 0; i < 16; i++) {
		W4_h[i] = __half(std::stof(tokens[i]));
		count++;
	}
	std::cout << "Set " << count << " values in W4" << std::endl;
	// Set values in b4
	std::string line4;
	std::getline(file4, line4);
	// std::cout << "line4: " << line4 << std::endl;
	b4_h[0] = __half(std::stof(line4));
	std::cout << "Set 1 value in b4" << std::endl;
	tokens.clear();
	file4.close();

	std::mt19937 gen(1996);
	std::uniform_real_distribution<float> dis(0.0, 1.0);

	for (auto &v : C_h) v = __half(0.0f);
	for (auto &v : D_h) v = __half(0.0f);
	// for (auto &v : x1_h) v = __half(dis(gen));
	for (auto &v : x1_h) v = __half(0.5f);
	// for (auto &v : x2_h) v = __half(0.5f);
	for (auto &v : x2_h) v = __half(G);
	for (auto &v : x3_h) v = __half(E);

	for (auto &v : A_h) v = __half(0.0f);
	for (auto &v : y_h) v = __half(0.0f);
	
	// for (int i = 0; i < N * K; i++) B_h[i] = (i % (K + 1) == 0) ? __half(1.0) : __half(0.0);
	// for (auto &v : B_h) v = __half(dis(gen));
	// for (auto &v : W11_h) v = __half(dis(gen));
	// for (auto &v : W12_h) v = __half(dis(gen));
	// for (auto &v : b1_h) v = __half(dis(gen));
	// for (auto &v : b2_h) v = __half(dis(gen));
	// for (auto &v : W3_h) v = __half(dis(gen));
	// b3_h[0] = __half(dis(gen));

	__dtype *A_d, *W2_d, *C_d, *D_d, *x1_d, *x2_d, *x3_d, *W11_d, *W12_d, *W13_d, *W3_d, *W4_d,
			*b1_d, *b2_d, *b3_d, *b4_d, *y_d;
	checkCudaStatus(cudaMalloc((void**)&A_d, sizeof(__dtype) * M * K));
	checkCudaStatus(cudaMalloc((void**)&W2_d, sizeof(__dtype) * K * N));
	checkCudaStatus(cudaMalloc((void**)&C_d, sizeof(__dtype) * M * N));
	checkCudaStatus(cudaMalloc((void**)&D_d, sizeof(__dtype) * M * N));
	checkCudaStatus(cudaMalloc((void**)&x1_d, sizeof(__dtype) * M));
	checkCudaStatus(cudaMalloc((void**)&x2_d, sizeof(__dtype) * M));
	checkCudaStatus(cudaMalloc((void**)&x3_d, sizeof(__dtype) * M));
	checkCudaStatus(cudaMalloc((void**)&W11_d, sizeof(__dtype) * K));
	checkCudaStatus(cudaMalloc((void**)&W12_d, sizeof(__dtype) * K));
	checkCudaStatus(cudaMalloc((void**)&W13_d, sizeof(__dtype) * K));
	checkCudaStatus(cudaMalloc((void**)&W3_d, sizeof(__dtype) * K * N));
	checkCudaStatus(cudaMalloc((void**)&W4_d, sizeof(__dtype) * N));
	checkCudaStatus(cudaMalloc((void**)&b1_d, sizeof(__dtype) * N));
	checkCudaStatus(cudaMalloc((void**)&b2_d, sizeof(__dtype) * N));
	checkCudaStatus(cudaMalloc((void**)&b3_d, sizeof(__dtype) * N));
	checkCudaStatus(cudaMalloc((void**)&b4_d, sizeof(__dtype) * 1));
	checkCudaStatus(cudaMalloc((void**)&y_d, sizeof(__dtype) * M));

	checkCudaStatus(cudaMemcpy(A_d, A_h.data(), sizeof(__dtype) * M * K, cudaMemcpyHostToDevice));
	// checkCudaStatus(cudaMemcpy(B_d, B_h.data(), sizeof(__dtype) * K * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W2_d, W2_h.data(), sizeof(__dtype) * K * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(C_d, C_h.data(), sizeof(__dtype) * M * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(D_d, D_h.data(), sizeof(__dtype) * M * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(x1_d, x1_h.data(), sizeof(__dtype) * M, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(x2_d, x2_h.data(), sizeof(__dtype) * M, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(x3_d, x3_h.data(), sizeof(__dtype) * M, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W11_d, W11_h.data(), sizeof(__dtype) * K, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W12_d, W12_h.data(), sizeof(__dtype) * K, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W13_d, W13_h.data(), sizeof(__dtype) * K, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W3_d, W3_h.data(), sizeof(__dtype) * K * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(W4_d, W4_h.data(), sizeof(__dtype) * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(b1_d, b1_h.data(), sizeof(__dtype) * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(b2_d, b2_h.data(), sizeof(__dtype) * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(b3_d, b3_h.data(), sizeof(__dtype) * N, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(b4_d, b4_h.data(), sizeof(__dtype) * 1, cudaMemcpyHostToDevice));
	checkCudaStatus(cudaMemcpy(y_d, y_h.data(), sizeof(__dtype) * M, cudaMemcpyHostToDevice));

	checkCudaStatus(cudaMemcpyToSymbol(W11_c, W11_h.data(), sizeof(__dtype) * K));
	checkCudaStatus(cudaMemcpyToSymbol(W12_c, W12_h.data(), sizeof(__dtype) * K));
	checkCudaStatus(cudaMemcpyToSymbol(W13_c, W13_h.data(), sizeof(__dtype) * K));
	checkCudaStatus(cudaMemcpyToSymbol(b1_c, b1_h.data(), sizeof(__dtype) * K));
	checkCudaStatus(cudaMemcpyToSymbol(b2_c, b2_h.data(), sizeof(__dtype) * N));
	// checkCudaStatus(cudaMemcpyToSymbol(W3_c, W3_h.data(), sizeof(__dtype) * K * N));
	checkCudaStatus(cudaMemcpyToSymbol(b3_c, b3_h.data(), sizeof(__dtype) * N));
	checkCudaStatus(cudaMemcpyToSymbol(W4_c, W4_h.data(), sizeof(__dtype) * N));
	checkCudaStatus(cudaMemcpyToSymbol(b4_c, b4_h.data(), sizeof(__dtype) * 1));

	// std::cout << "Starting cpu computation" << std::endl;
	if(cpu_comparison){
		for (int i=0; i < M; i++){
			for (int j=0; j < K; j++){
				A_h[i * K + j] = activ(x1_h[i] * W11_h[j]+ x2_h[i] * W12_h[j] + x3_h[i] * W13_h[j] + b1_h[j]);
			}
		}
		// C[M, N] = A[M, K] * B[K, N] + b2[1, N]
		for (int i=0; i < M; i++){
			for (int j=0; j < N; j++){
				for (int k=0; k < K; k++){
					C_h[i * N + j] += A_h[i * K + k] * W2_h[k * N + j];
				}
				C_h[i * N + j] = activ(C_h[i * N + j] + b2_h[j]);
				// C_h[i * N + j] = activ(C_h[i * N + j]);
			}
		}
		// D[M, N] = C[M, K] * W3[K, N] + b3[1, N]
		for (int i=0; i < M; i++){
			for (int j=0; j < N; j++){
				for (int k=0; k < K; k++){
					D_h[i * N + j] += C_h[i * N + k] * W3_h[k * N + j];
				}
				D_h[i * N + j] = activ(D_h[i * N + j] + b3_h[j]);
			}
		}

		// y[M, 1] = C[M, N] * W3[N, 1] + b3[1, 1]
		for (int i=0; i < M; i++){
			for (int j=0; j < N; j++){
				y_h[i] += D_h[i * N + j] * W4_h[j];
			}
			y_h[i] = hardtanh(y_h[i] + b4_h[0]);
		}
	}

	int num_reps = 1;
	auto start = std::chrono::high_resolution_clock::now();
	// void wmmaCall(half *A, half *W2, half *C, half *D,
	// 		  half *x1, half *x2, half *x3,
	// 		  half *W11, half *W12, half *W13, half *W3, half *W4,
	// 		  half *b1, half *b2, half *b3, half *b4, half *y,
	// 		  size_t M, size_t N, size_t K) {
	for(int i = 0; i < num_reps; i++) {
		wmmaCall(A_d, W2_d, C_d, D_d,
				 x1_d, x2_d, x3_d, 
				 W11_d, W12_d, W13_d, W3_d, W4_d,
				 b1_d, b2_d, b3_d, b4_d, y_d,
				 M, N, K);
	}
	checkCudaStatus(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Elapsed time: " << elapsed.count() / num_reps << " s" << std::endl;
	std::cout << "Throughput: " << (double)M / (elapsed.count() / num_reps) << " particles per second, ";
	std::cout << (double)M / (elapsed.count() / num_reps) *1e-9 << " per ns" << std::endl;
	std::cout << "Time per particle: " << (elapsed.count() / num_reps) / M * 1e9 << " ns" << std::endl;

	if(cpu_comparison){
		// checkCudaStatus(cudaMemcpy(C_h.data(), C_d, sizeof(__dtype) * M * N, cudaMemcpyDeviceToHost));
		checkCudaStatus(cudaMemcpy(y_out_h.data(), y_d, sizeof(__dtype) * M, cudaMemcpyDeviceToHost));
		__half tol = 0.005;
		compare(y_out_h.data(), y_h.data(), M, tol);
		// compare(C_h.data(), A_h.data(), M * N, tol);
	}

	if(!cpu_comparison){
		checkCudaStatus(cudaMemcpy(y_out_h.data(), y_d, sizeof(__dtype) * M, cudaMemcpyDeviceToHost));
	}

	if(write_output){
		std::ofstream outfile;
		outfile.open(outputFileName);
		std::cout << "Writing to file: " << outputFileName << std::endl;
		outfile << float(y_out_h[0]);
		for (int i = 1; i < M; i++) {
			outfile << ", " << float(y_out_h[i]);
		}
		outfile.close();
	}

	return 0;
}
