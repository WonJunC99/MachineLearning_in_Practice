#pragma once
#include "Imagelib.h"
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
	Tensor3D* cached_output; // backward에서 ReLU 마스크 계산에 사용 (forward 출력값 저장)
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out)
	{
		// (구현할 것)
		// 동작1: Base class의 생성자를 호출하여 맴버 변수를 초기화 할 것(반드시 initialization list를 사용할 것)
		// 동작2: cached_output을 nullptr로 초기화 할 것
	}
	~Layer_ReLU() {
		// (구현할 것)
		// 동작: cached_output이 nullptr이 아니면 동적 할당 해제
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		// 동작1: input tensor에 대해 각 element x가 양수이면 그대로 전달, 음수이면 0으로 output tensor에 전달
		// 동작2: output tensor는 동적할당하여 주소값을 반환
		// 동작3: backward에서 마스크로 사용할 수 있도록 output을 cached_output에 복사하여 저장
		//        (cached_output이 기존에 할당되어 있으면 해제 후 새로 할당할 것)
		// 함수:  Tensor3D의 get_info(), get_elem(), set_elem()을 적절히 활용할 것
		// [OpenMP] clock()으로 병렬처리 전/후 처리속도를 측정하고, #pragma omp parallel for를
		//          적용하여 루프를 병렬처리 할 것 (적절한 루프에 적용할 것)

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
	}
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		// 동작: 맴버 변수들을 pass by reference로 외부에 전달
	}
	void print() const override {
		// (구현할 것)
		// 동작: layer 이름 및 크기 정보를 화면에 출력
	}
};


class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double* weight_tensor;      // fK x fK x fC_in x fC_out 연속 메모리: 인덱스 = ((ph*fK+pw)*fC_in+ci)*fC_out+co
	double* bias_tensor;        // fC_out 크기의 1차원 배열 (bias는 각 filter당 1개)

	Tensor3D* cached_input;     // backward에서 dL/dW 계산에 사용 (forward 입력값 저장)
	double* grad_weight_tensor; // 가중치의 gradient (weight_tensor와 동일한 레이아웃)
	double* grad_bias_tensor;   // bias의 gradient (fC_out)
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "") {
		// (구현할 것)
		// 동작1: initialization list와 base class의 생성자를 이용하여 맴버 변수를 초기화 할 것
		// 동작2: filename_weight와 filename_bias를 저장하고, cached_input은 nullptr로 초기화
		// 동작3: new double[fK*fK*fC_in*fC_out]()와 new double[fC_out]()를 사용하여
		//        weight_tensor, bias_tensor, grad_weight_tensor, grad_bias_tensor를 동적 할당할 것
		//        (끝의 ()는 0으로 초기화, weight 인덱스 = ((ph*fK+pw)*fC_in+ci)*fC_out+co)
		// 동작4: init() 함수를 호출하여 가중치를 초기화 할 것

		init(init_type);
	}
	void init(int init_type) {
		// (구현할 것)
		// 동작1: init_type (MEAN_INIT 또는 LOAD_INIT)에 따라 가중치를 다른 방식으로 초기화
		// 동작2: MEAN_INIT - 모든 가중치를 1/(fK*fK*fC_in)으로, bias는 0으로 초기화
		// 동작3: LOAD_INIT - filename_weight, filename_bias 파일에서 값을 읽어 가중치에 저장
		//        (파일에서 읽는 순서: 필터, 채널, 행, 열 순서로 채워짐)
	}
	~Layer_Conv() override {
		// (구현할 것)
		// 동작: weight_tensor, bias_tensor, grad_weight_tensor, grad_bias_tensor를 할당 해제
		//       cached_input이 nullptr이 아니면 동적 할당 해제
		// 함수: delete[] 사용 (flat 배열이므로 free_dmatrix4D 불필요)
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		// 동작1: 컨볼루션 (각 위치마다 y = WX + b)를 수행
		// 동작2: output (Tensor3D type)를 먼저 동적 할당하고 연산이 완료된 다음 pointer를 반환
		// 동작3: backward에서 dL/dW 계산에 사용하기 위해 input을 cached_input에 복사하여 저장
		//        (cached_input이 기존에 할당되어 있으면 해제 후 새로 할당할 것)
		// [OpenMP] clock()으로 병렬처리 전/후 처리속도를 측정하고, #pragma omp parallel for를
		//          적용하여 컨볼루션 루프를 병렬처리 할 것 (적절한 루프에 적용할 것)

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
	}
	void update_weights(double lr) override {
		// (구현할 것)
		// SGD(Stochastic Gradient Descent) weight 업데이트
		// 동작: W  = W  - lr * grad_weight_tensor
		//       b  = b  - lr * grad_bias_tensor
		// [OpenMP] #pragma omp parallel for를 적용하여 병렬처리 할 것
	}
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		// 동작: Layer_ReLU와 동일
	}
	void print() const override {
		// (구현할 것)
		// 동작: Layer_ReLU와 동일
	}
};
