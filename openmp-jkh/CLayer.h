#pragma once
#include "ImageLib.h"
#include "CTensor.h"
#include <omp.h>   // OpenMP 병렬처리
#include <ctime>   // clock()으로 처리속도 측정

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer는 tensor를 입/출력으로 가지며, 특정 operation을 수행하는 Convolutional Neural Network의 기본 연산 단위

class Layer {
protected:
	int fK;     // kernel size in K*K kernel
	int fC_in;  // number of input channels
	int fC_out; // number of filters (output channels)
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; // 가상소멸자 (참고: https://wonjayk.tistory.com/243)
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	virtual Tensor3D* backward(const Tensor3D* grad_output) = 0;
	virtual void update_weights(double lr) {} // 가중치가 없는 layer(ReLU)는 기본 동작 없음
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
private:
	Tensor3D* cached_output; // backward에서 ReLU 마스크 계산에 사용 (forward 출력 tensor를 non-owning으로 저장)
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out)
		: Layer(_name, _fK, _fC_in, _fC_out), cached_output(nullptr)
	{
		// (구현할 것)
		// 동작1: Base class의 생성자를 호출하여 맴버 변수를 초기화 할 것(반드시 initialization list를 사용할 것)
		// 동작2: cached_output을 nullptr로 초기화 할 것
	}
	~Layer_ReLU() {
		// (구현할 것)
		// 동작: cached_output은 Model/tensor vector가 관리하는 non-owning 포인터이므로 해제하지 않음
		cached_output = nullptr;
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		// 동작1: input tensor에 대해 각 element x가 양수이면 그대로 전달, 음수이면 0으로 output tensor에 전달
		// 동작2: output tensor는 동적할당하여 주소값을 반환
		// 동작3: backward에서 마스크로 사용할 수 있도록 output 주소를 cached_output에 저장
		//        output tensor는 Model/tensor vector 또는 호출자가 관리하므로 cached_output은 delete하지 않음
		// 함수:  Tensor3D의 get_info(), get_elem(), set_elem()을 적절히 활용할 것
		// [OpenMP] clock()으로 병렬처리 전/후 처리속도를 측정하고, #pragma omp parallel for를
		//          적용하여 루프를 병렬처리 할 것 (적절한 루프에 적용할 것)

		int nH, nW, nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, nC);

		double start = omp_get_wtime();
		int total = nH * nW * nC;
		const double* input_data = input->tensor;
		double* output_data = output->tensor;
#pragma omp parallel for schedule(static)
		for (int i = 0; i < total; i++) {
			output_data[i] = (input_data[i] > 0.0) ? input_data[i] : 0.0;
		}

		cached_output = output;

		double end = omp_get_wtime();
		double elapsed_time = end - start;
		cout << "Processing time: " << elapsed_time << "s" << endl;
		cout << name << " is finished" << endl;
		return output;
	}
	Tensor3D* backward(const Tensor3D* grad_output) override {
		// (구현할 것)
		// 동작: ReLU backward - forward에서 출력이 양수였던 위치만 gradient를 통과시킴
		// 수식: dL/dX[h][w][c] = grad_output[h][w][c]  if cached_output[h][w][c] > 0
		//                       = 0                     otherwise
		// 반환: grad_input (동적할당 후 반환)
		// [OpenMP] forward와 동일하게 #pragma omp parallel for를 적용하여 루프를 병렬처리 할 것
		int nH, nW, nC;
		grad_output->get_info(nH, nW, nC);
		Tensor3D* grad_input = new Tensor3D(nH, nW, nC);

		int total = nH * nW * nC;
		const double* grad_output_data = grad_output->tensor;
		const double* cache_data = cached_output->tensor;
		double* grad_input_data = grad_input->tensor;
#pragma omp parallel for schedule(static)
		for (int i = 0; i < total; i++) {
			grad_input_data[i] = (cache_data[i] > 0.0) ? grad_output_data[i] : 0.0;
		}
		return grad_input;
	}
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		// 동작: 맴버 변수들을 pass by reference로 외부에 전달
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (구현할 것)
		// 동작: layer 이름 및 크기 정보를 화면에 출력
		cout << name << " [" << fK << "x" << fK << "x" << fC_in << "x" << fC_out << "]" << endl;
	}
};


class Layer_Conv : public Layer {
private:
	friend class Model;
	string filename_weight;
	string filename_bias;
	double* weight_tensor;      // fK x fK x fC_in x fC_out 연속 메모리: 인덱스 = ((ph*fK+pw)*fC_in+ci)*fC_out+co
	double* bias_tensor;        // fC_out 크기의 1차원 배열 (bias는 각 filter당 1개)

	Tensor3D* cached_input;     // backward에서 dL/dW 계산에 사용 (forward 입력 tensor를 non-owning으로 저장)
	double* grad_weight_tensor; // 가중치의 gradient (weight_tensor와 동일한 레이아웃)
	double* grad_bias_tensor;   // bias의 gradient (fC_out)
	bool compute_grad_input;    // Model::train_step에서 첫 layer의 불필요한 dL/dX 계산을 생략하기 위한 flag
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "")
		: Layer(_name, _fK, _fC_in, _fC_out),
		filename_weight(_filename_weight), filename_bias(_filename_bias),
		weight_tensor(nullptr), bias_tensor(nullptr), cached_input(nullptr),
		grad_weight_tensor(nullptr), grad_bias_tensor(nullptr), compute_grad_input(true)
	{
		// (구현할 것)
		// 동작1: initialization list와 base class의 생성자를 이용하여 맴버 변수를 초기화 할 것
		// 동작2: filename_weight와 filename_bias를 저장하고, cached_input은 nullptr로 초기화
		// 동작3: new double[fK*fK*fC_in*fC_out]()와 new double[fC_out]()를 사용하여
		//        weight_tensor, bias_tensor, grad_weight_tensor, grad_bias_tensor를 동적 할당할 것
		//        (끝의 ()는 0으로 초기화, weight 인덱스 = ((ph*fK+pw)*fC_in+ci)*fC_out+co)
		// 동작4: init() 함수를 호출하여 가중치를 초기화 할 것

		int weight_size = fK * fK * fC_in * fC_out;
		weight_tensor = new double[weight_size]();
		bias_tensor = new double[fC_out]();
		grad_weight_tensor = new double[weight_size]();
		grad_bias_tensor = new double[fC_out]();
		init(init_type);
	}
	void init(int init_type) {
		// (구현할 것)
		// 동작1: init_type (MEAN_INIT 또는 LOAD_INIT)에 따라 가중치를 다른 방식으로 초기화
		// 동작2: MEAN_INIT - 모든 가중치를 1/(fK*fK*fC_in)으로, bias는 0으로 초기화
		// 동작3: LOAD_INIT - filename_weight, filename_bias 파일에서 값을 읽어 가중치에 저장
		//        (파일에서 읽는 순서: 필터, 채널, 행, 열 순서로 채워짐)
		int weight_size = fK * fK * fC_in * fC_out;
		if (init_type == MEAN_INIT) {
			double val = 1.0 / (double)(fK * fK * fC_in);
			for (int i = 0; i < weight_size; i++) weight_tensor[i] = val;
			for (int co = 0; co < fC_out; co++) bias_tensor[co] = 0.0;
		}
		else if (init_type == LOAD_INIT) {
			ifstream fin_weight(filename_weight);
			for (int co = 0; co < fC_out; co++) {
				for (int ci = 0; ci < fC_in; ci++) {
					for (int ph = 0; ph < fK; ph++) {
						for (int pw = 0; pw < fK; pw++) {
							double val = 0.0;
							fin_weight >> val;
							int idx = ((ph * fK + pw) * fC_in + ci) * fC_out + co;
							weight_tensor[idx] = val;
						}
					}
				}
			}
			fin_weight.close();

			ifstream fin_bias(filename_bias);
			for (int co = 0; co < fC_out; co++) {
				double val = 0.0;
				fin_bias >> val;
				bias_tensor[co] = val;
			}
			fin_bias.close();
		}
	}
	~Layer_Conv() override {
		// (구현할 것)
		// 동작: weight_tensor, bias_tensor, grad_weight_tensor, grad_bias_tensor를 할당 해제
		//       cached_input은 Model/tensor vector가 관리하는 non-owning 포인터이므로 해제하지 않음
		// 함수: delete[] 사용 (flat 배열이므로 free_dmatrix4D 불필요)
		delete[] weight_tensor;
		delete[] bias_tensor;
		delete[] grad_weight_tensor;
		delete[] grad_bias_tensor;
		cached_input = nullptr;
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		// 동작1: 컨볼루션 (각 위치마다 y = WX + b)를 수행
		// 동작2: output (Tensor3D type)를 먼저 동적 할당하고 연산이 완료된 다음 pointer를 반환
		// 동작3: backward에서 dL/dW 계산에 사용하기 위해 input tensor의 주소를 cached_input에 저장
		//        input tensor는 Model/tensor vector가 관리하므로 cached_input은 delete하지 않음
		// [OpenMP] clock()으로 병렬처리 전/후 처리속도를 측정하고, #pragma omp parallel for를
		//          적용하여 컨볼루션 루프를 병렬처리 할 것 (적절한 루프에 적용할 것)

		int nH, nW, nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, fC_out);

		cached_input = const_cast<Tensor3D*>(input);

		double start = omp_get_wtime();
		int offset = fK / 2;
		int h_begin = offset;
		int h_end = nH - offset;
		int w_begin = offset;
		int w_end = nW - offset;
		int validH = h_end - h_begin;
		int validW = w_end - w_begin;
		if (validH < 0) validH = 0;
		if (validW < 0) validW = 0;
		int total_pixels = validH * validW;

		const double* input_data = input->tensor;
		double* output_data = output->tensor;

		if (fC_out == 1) {
#pragma omp parallel for collapse(2) schedule(static)
			for (int h = h_begin; h < h_end; h++) {
				for (int w = w_begin; w < w_end; w++) {
					double sum = bias_tensor[0];
					for (int ph = 0; ph < fK; ph++) {
						int ih = h + ph - offset;
						for (int pw = 0; pw < fK; pw++) {
							int iw = w + pw - offset;
							int input_base = (ih * nW + iw) * nC;
							int weight_base = (ph * fK + pw) * fC_in;
#pragma omp simd reduction(+:sum)
							for (int ci = 0; ci < fC_in; ci++) {
								sum += weight_tensor[weight_base + ci] * input_data[input_base + ci];
							}
						}
					}
					output_data[h * nW + w] = sum;
				}
			}
		}
		else if (fC_out <= 128) {
			int blockW = (validW + 3) / 4;
#pragma omp parallel for collapse(2) schedule(static)
			for (int h = h_begin; h < h_end; h++) {
				for (int wb = 0; wb < blockW; wb++) {
					double s0[128], s1[128], s2[128], s3[128];
					int w = w_begin + wb * 4;

					if (w + 3 < w_end) {
#pragma omp simd
						for (int co = 0; co < fC_out; co++) {
							double b = bias_tensor[co];
							s0[co] = b;
							s1[co] = b;
							s2[co] = b;
							s3[co] = b;
						}

						for (int ph = 0; ph < fK; ph++) {
							int ih = h + ph - offset;
							for (int pw = 0; pw < fK; pw++) {
								int iw = w + pw - offset;
								int input_base = (ih * nW + iw) * nC;
								int weight_base = ((ph * fK + pw) * fC_in) * fC_out;
								for (int ci = 0; ci < fC_in; ci++) {
									double x0 = input_data[input_base + ci];
									double x1 = input_data[input_base + nC + ci];
									double x2 = input_data[input_base + 2 * nC + ci];
									double x3 = input_data[input_base + 3 * nC + ci];
									int w_idx = weight_base + ci * fC_out;
#pragma omp simd
									for (int co = 0; co < fC_out; co++) {
										double wt = weight_tensor[w_idx + co];
										s0[co] += x0 * wt;
										s1[co] += x1 * wt;
										s2[co] += x2 * wt;
										s3[co] += x3 * wt;
									}
								}
							}
						}

						int out0 = (h * nW + w) * fC_out;
						int out1 = out0 + fC_out;
						int out2 = out1 + fC_out;
						int out3 = out2 + fC_out;
#pragma omp simd
						for (int co = 0; co < fC_out; co++) {
							output_data[out0 + co] = s0[co];
							output_data[out1 + co] = s1[co];
							output_data[out2 + co] = s2[co];
							output_data[out3 + co] = s3[co];
						}
					}
					else {
						for (; w < w_end; w++) {
#pragma omp simd
							for (int co = 0; co < fC_out; co++) s0[co] = bias_tensor[co];

							for (int ph = 0; ph < fK; ph++) {
								int ih = h + ph - offset;
								for (int pw = 0; pw < fK; pw++) {
									int iw = w + pw - offset;
									int input_base = (ih * nW + iw) * nC;
									int weight_base = ((ph * fK + pw) * fC_in) * fC_out;
									for (int ci = 0; ci < fC_in; ci++) {
										double x = input_data[input_base + ci];
										int w_idx = weight_base + ci * fC_out;
#pragma omp simd
										for (int co = 0; co < fC_out; co++) s0[co] += x * weight_tensor[w_idx + co];
									}
								}
							}

							int output_base = (h * nW + w) * fC_out;
#pragma omp simd
							for (int co = 0; co < fC_out; co++) output_data[output_base + co] = s0[co];
						}
					}
				}
			}
		}
		else {
#pragma omp parallel
			{
				vector<double> sums(fC_out);
#pragma omp for collapse(2) schedule(static)
				for (int h = h_begin; h < h_end; h++) {
					for (int w = w_begin; w < w_end; w++) {
#pragma omp simd
						for (int co = 0; co < fC_out; co++) sums[co] = bias_tensor[co];

						for (int ph = 0; ph < fK; ph++) {
							int ih = h + ph - offset;
							for (int pw = 0; pw < fK; pw++) {
								int iw = w + pw - offset;
								int input_base = (ih * nW + iw) * nC;
								int weight_base = ((ph * fK + pw) * fC_in) * fC_out;
								for (int ci = 0; ci < fC_in; ci++) {
									double x = input_data[input_base + ci];
									int w_idx = weight_base + ci * fC_out;
#pragma omp simd
									for (int co = 0; co < fC_out; co++) sums[co] += x * weight_tensor[w_idx + co];
								}
							}
						}

						int output_base = (h * nW + w) * fC_out;
#pragma omp simd
						for (int co = 0; co < fC_out; co++) output_data[output_base + co] = sums[co];
					}
				}
			}
		}

		double end = omp_get_wtime();
		double elapsed_time = end - start;
		cout << "Processing time: " << elapsed_time << "s" << endl;
		cout << name << " is finished" << endl;
		return output;
	}
	Tensor3D* backward(const Tensor3D* grad_output) override {
		// (구현할 것)
		// Convolution backward pass: dL/dX, dL/dW, dL/db를 각각 계산
		//
		// [수식 정리]
		// forward:  Y[h][w][c_out] = sum_{ph,pw,c_in} W[ph][pw][c_in][c_out]
		//                            * X[h+ph-offset][w+pw-offset][c_in] + b[c_out]
		//
		// dL/db[c_out]                 += grad_output[h][w][c_out]
		// dL/dW[ph][pw][c_in][c_out]   += grad_output[h][w][c_out] * X[h+ph-offset][w+pw-offset][c_in]
		// dL/dX[h+ph-offset][w+pw-offset][c_in] += grad_output[h][w][c_out] * W[ph][pw][c_in][c_out]
		//
		// 참고: W[ph][pw][c_in][c_out] = weight_tensor[((ph*fK+pw)*fC_in+c_in)*fC_out+c_out] (flat 인덱스)
		//       grad_weight_tensor도 동일한 인덱스 사용
		//
		// 동작1: grad_weight_tensor, grad_bias_tensor를 0으로 초기화
		// 동작2: 유효한 출력 위치(offset ~ nH-offset)를 순회하며 위 수식으로 gradient 누적
		// 동작3: grad_input (dL/dX)을 동적 할당하여 반환
		// [OpenMP] forward와 동일하게 #pragma omp parallel for를 적용하여 루프를 병렬처리 할 것
		//          주의: dL/dX, dL/dW 누적 연산 시 race condition이 발생하지 않도록 할 것
		int nH, nW, nC;
		cached_input->get_info(nH, nW, nC);

		Tensor3D* grad_input = new Tensor3D(nH, nW, fC_in);
		int weight_size = fK * fK * fC_in * fC_out;
		int input_size = nH * nW * fC_in;

		int offset = fK / 2;
		int h_begin = offset;
		int h_end = nH - offset;
		int w_begin = offset;
		int w_end = nW - offset;
		int validH = h_end - h_begin;
		int validW = w_end - w_begin;
		if (validH < 0) validH = 0;
		if (validW < 0) validW = 0;
		int valid_pixels = validH * validW;
		int kernel_channels = fK * fK * fC_in;

		const double* input_data = cached_input->tensor;
		const double* grad_output_data = grad_output->tensor;
		double* grad_input_data = grad_input->tensor;

		long long grad_weight_work = (long long)valid_pixels * kernel_channels * fC_out;
		int num_threads = omp_get_max_threads();
		if (num_threads < 1) num_threads = 1;
		vector< vector<double> > local_grad_weight(num_threads, vector<double>(weight_size, 0.0));
		vector< vector<double> > local_grad_bias(num_threads, vector<double>(fC_out, 0.0));

#pragma omp parallel num_threads(num_threads)
		{
			int tid = omp_get_thread_num();
			double* l_grad_weight = local_grad_weight[tid].data();
			double* l_grad_bias = local_grad_bias[tid].data();

#pragma omp for collapse(2) schedule(static)
			for (int h = h_begin; h < h_end; h++) {
				for (int w = w_begin; w < w_end; w++) {
					int grad_base = (h * nW + w) * fC_out;

#pragma omp simd
					for (int co = 0; co < fC_out; co++) {
						l_grad_bias[co] += grad_output_data[grad_base + co];
					}

					for (int ph = 0; ph < fK; ph++) {
						int ih = h + ph - offset;
						for (int pw = 0; pw < fK; pw++) {
							int iw = w + pw - offset;
							int input_base = (ih * nW + iw) * fC_in;
							int weight_base = ((ph * fK + pw) * fC_in) * fC_out;

							for (int ci = 0; ci < fC_in; ci++) {
								double x = input_data[input_base + ci];
								int w_idx = weight_base + ci * fC_out;
#pragma omp simd
								for (int co = 0; co < fC_out; co++) {
									l_grad_weight[w_idx + co] += grad_output_data[grad_base + co] * x;
								}
							}
						}
					}
				}
			}
		}

#pragma omp parallel for schedule(static)
		for (int i = 0; i < weight_size; i++) {
			double sum = 0.0;
			for (int t = 0; t < num_threads; t++) sum += local_grad_weight[t][i];
			grad_weight_tensor[i] = sum;
		}

#pragma omp parallel for schedule(static)
		for (int co = 0; co < fC_out; co++) {
			double sum = 0.0;
			for (int t = 0; t < num_threads; t++) sum += local_grad_bias[t][co];
			grad_bias_tensor[co] = sum;
		}

		if (!compute_grad_input) {
			compute_grad_input = true;
			return grad_input;
		}

		// input gradient: 입력 element 하나를 한 thread가 전담하므로 dL/dX 누적 race가 없다.
		// ci 방향 4-way register blocking으로 같은 grad_output 값을 여러 input channel 계산에 재사용한다.
		if (fC_out == 1) {
#pragma omp parallel for collapse(2) schedule(static)
			for (int h = 0; h < nH; h++) {
				for (int w = 0; w < nW; w++) {
					int ph_start = h + offset - h_end + 1;
					if (ph_start < 0) ph_start = 0;
					int ph_end = h + offset - h_begin;
					if (ph_end > fK - 1) ph_end = fK - 1;
					int pw_start = w + offset - w_end + 1;
					if (pw_start < 0) pw_start = 0;
					int pw_end = w + offset - w_begin;
					if (pw_end > fK - 1) pw_end = fK - 1;

					int out_base = (h * nW + w) * fC_in;
					int ci = 0;
					for (; ci + 3 < fC_in; ci += 4) {
						double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
						for (int ph = ph_start; ph <= ph_end; ph++) {
							int oh = h - ph + offset;
							for (int pw = pw_start; pw <= pw_end; pw++) {
								int ow = w - pw + offset;
								double go = grad_output_data[oh * nW + ow];
								int weight_idx = (ph * fK + pw) * fC_in + ci;
								sum0 += go * weight_tensor[weight_idx];
								sum1 += go * weight_tensor[weight_idx + 1];
								sum2 += go * weight_tensor[weight_idx + 2];
								sum3 += go * weight_tensor[weight_idx + 3];
							}
						}
						grad_input_data[out_base + ci] = sum0;
						grad_input_data[out_base + ci + 1] = sum1;
						grad_input_data[out_base + ci + 2] = sum2;
						grad_input_data[out_base + ci + 3] = sum3;
					}
					for (; ci < fC_in; ci++) {
						double sum = 0.0;
						for (int ph = ph_start; ph <= ph_end; ph++) {
							int oh = h - ph + offset;
							for (int pw = pw_start; pw <= pw_end; pw++) {
								int ow = w - pw + offset;
								int weight_idx = (ph * fK + pw) * fC_in + ci;
								sum += grad_output_data[oh * nW + ow] * weight_tensor[weight_idx];
							}
						}
						grad_input_data[out_base + ci] = sum;
					}
				}
			}
		}
		else {
#pragma omp parallel for collapse(2) schedule(static)
			for (int h = 0; h < nH; h++) {
				for (int w = 0; w < nW; w++) {
					int ph_start = h + offset - h_end + 1;
					if (ph_start < 0) ph_start = 0;
					int ph_end = h + offset - h_begin;
					if (ph_end > fK - 1) ph_end = fK - 1;
					int pw_start = w + offset - w_end + 1;
					if (pw_start < 0) pw_start = 0;
					int pw_end = w + offset - w_begin;
					if (pw_end > fK - 1) pw_end = fK - 1;

					int out_base = (h * nW + w) * fC_in;
					int ci = 0;
					for (; ci + 3 < fC_in; ci += 4) {
						double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
						for (int ph = ph_start; ph <= ph_end; ph++) {
							int oh = h - ph + offset;
							for (int pw = pw_start; pw <= pw_end; pw++) {
								int ow = w - pw + offset;
								int grad_base = (oh * nW + ow) * fC_out;
								int weight_base = ((ph * fK + pw) * fC_in + ci) * fC_out;
								const double* w0 = weight_tensor + weight_base;
								const double* w1 = w0 + fC_out;
								const double* w2 = w1 + fC_out;
								const double* w3 = w2 + fC_out;
#pragma omp simd reduction(+:sum0,sum1,sum2,sum3)
								for (int co = 0; co < fC_out; co++) {
									double go = grad_output_data[grad_base + co];
									sum0 += go * w0[co];
									sum1 += go * w1[co];
									sum2 += go * w2[co];
									sum3 += go * w3[co];
								}
							}
						}
						grad_input_data[out_base + ci] = sum0;
						grad_input_data[out_base + ci + 1] = sum1;
						grad_input_data[out_base + ci + 2] = sum2;
						grad_input_data[out_base + ci + 3] = sum3;
					}
					for (; ci < fC_in; ci++) {
						double sum = 0.0;
						for (int ph = ph_start; ph <= ph_end; ph++) {
							int oh = h - ph + offset;
							for (int pw = pw_start; pw <= pw_end; pw++) {
								int ow = w - pw + offset;
								int grad_base = (oh * nW + ow) * fC_out;
								int weight_base = ((ph * fK + pw) * fC_in + ci) * fC_out;
#pragma omp simd reduction(+:sum)
								for (int co = 0; co < fC_out; co++) {
									sum += grad_output_data[grad_base + co] * weight_tensor[weight_base + co];
								}
							}
						}
						grad_input_data[out_base + ci] = sum;
					}
				}
			}
		}

		return grad_input;
	}
	void update_weights(double lr) override {
		// (구현할 것)
		// SGD(Stochastic Gradient Descent) weight 업데이트
		// 동작: W  = W  - lr * grad_weight_tensor
		//       b  = b  - lr * grad_bias_tensor
		// [OpenMP] #pragma omp parallel for를 적용하여 병렬처리 할 것
		int weight_size = fK * fK * fC_in * fC_out;
#pragma omp parallel for schedule(static)
		for (int i = 0; i < weight_size; i++) {
			weight_tensor[i] -= lr * grad_weight_tensor[i];
		}
#pragma omp parallel for schedule(static)
		for (int co = 0; co < fC_out; co++) {
			bias_tensor[co] -= lr * grad_bias_tensor[co];
		}
	}
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		// 동작: Layer_ReLU와 동일
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (구현할 것)
		// 동작: Layer_ReLU와 동일
		cout << name << " [" << fK << "x" << fK << "x" << fC_in << "x" << fC_out << "]" << endl;
	}
};
