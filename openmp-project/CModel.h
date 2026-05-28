#pragma once
#include "CLayer.h"
#include "CLoss.h"

// Model은 layer와 tensor들을 모두 통합 관리하여 효과적으로 CNN이 수행될 수 있도록 함

class Model {
private:
	vector<Layer*> layers;    // layer들을 순차적으로 저장
	vector<Tensor3D*> tensors;// tensor들을 순차적으로 저장
	                          // ( 0번째 tensor는 0번째 layer의 입력,
	                          //   i번째 tensor는 i번째 layer의 입력이자 (i-1)번째 layer의 출력 )
public:
	Model() {}
	void add_layer(Layer* layer) {
		// (구현할 것)
		// 동작: layer 객체를 layers vector의 마지막 element로 저장
		layers.push_back(layer);
	}
	~Model() {
		// (구현할 것)
		// 동작: layers와 tensors의 모든 element를 동적할당 해제
		for (auto l : layers) delete l;
		for (auto t : tensors) delete t;
	}

	// =========================================================================
	// forward pass (inference)
	// =========================================================================
	void test(string filename_input, string filename_output) {
		// (구현할 것)
		// 동작1: filename_input으로부터 이미지를 읽어와서 tensor로 변환한 다음
		//        CNN을 수행하고 그 결과물을 filename_output에 저장
		// 동작2: 아래 주석 (1)~(4) 중 (2)번만 구현하면 됨

		// 이전 tensors 초기화
		for (auto t : tensors) delete t;
		tensors.clear();

		int nH, nW;
		double** input_img_Y, **input_img_U, **input_img_V;
		byte* pLoadImage;

		// (1) 영상을 읽어서 2차원 배열로 저장 (input_img_Y, U, V는 read_image에서 동적 할당됨)
		read_image(filename_input, pLoadImage, input_img_Y, input_img_U, input_img_V, nH, nW);
		cout << "Reading (" << filename_input << ") is complete..." << endl;

		// (2) 이 부분만 구현할 것
		// 동작1: tensors의 0번째 element(입력 tensor)가 이미 read_image에서 저장되어 있음
		// 동작2: tensors의 i번째 tensor를 layers의 i번째 layer의 forward 함수 입력으로 넣고,
		//        그 결과를 tensors의 (i+1)번째 tensor로 저장
		// 동작3: 결과적으로 tensors의 가장 마지막 tensor가 CNN의 최종 출력값이 됨
		for (int i = 0; i < (int)layers.size(); i++) {
    		Tensor3D* output = layers[i]->forward(tensors[i]);
    		tensors.push_back(output);
		}
		cout << "Super-resolution is complete..." << endl;

		// (3) CNN의 출력(마지막 tensor)을 2차원 배열로 변환 후 U, V 채널과 함께 이미지로 저장
		Tensor3D* output_tensor_Y = tensors.at(tensors.size() - 1);
		save_image(filename_output, pLoadImage, output_tensor_Y, input_img_U, input_img_V, nH, nW);
		cout << "Saving (" << filename_output << ") is complete..." << endl;

		// (4) 할당 해제
		free(pLoadImage);
		free_dmatrix2D(input_img_Y, nH, nW);
		free_dmatrix2D(input_img_U, nH, nW);
		free_dmatrix2D(input_img_V, nH, nW);
	}

	// =========================================================================
	// backward pass (training)
	// =========================================================================
	void train(string filename_input, string filename_target, double lr, int epochs = 100) {
		// (구현할 것)
		// 동작1: filename_input(저해상도)와 filename_target(고해상도)을 읽어서 각각 tensor로 변환
		//        - read_image()로 입력 이미지 로드 (tensors[0]에 저장됨)
		//        - LoadBmp/convert 함수로 target 이미지의 Y채널(double**)을 얻은 뒤
		//          new Tensor3D(nH, nW, 1)을 생성하고 set_elem()으로 값을 채울 것 (target은 별도 관리)
		// 동작2: epochs 횟수만큼 아래 루프를 반복
		//        - cout << "[Epoch " << e+1 << "/" << epochs << "]" << endl; 출력
		//        - train_step(tensors[0], target, lr) 호출
		// 동작3: 사용한 메모리 모두 할당 해제 (pLoadImage, pTargetImage, img 배열, target tensor 등)
		// 주의: target tensor는 루프 밖에서 한 번만 생성할 것
		for (auto t : tensors) delete t;
		tensors.clear();
    	
		// 입력 이미지 로드 (tensors[0]에 저장됨)
    	int nH, nW;
    	double** input_img_Y, **input_img_U, **input_img_V;
    	byte* pLoadImage;
    	read_image(filename_input, pLoadImage, input_img_Y, input_img_U, input_img_V, nH, nW);

		// 타겟 이미지 로드
    	int nH_t, nW_t;
	    byte* pTargetImage;
    	double** target_img_Y = dmatrix2D(nH, nW);
    	double** target_img_U = dmatrix2D(nH, nW);
    	double** target_img_V = dmatrix2D(nH, nW);
    	LoadBmp(filename_target.c_str(), &pTargetImage, nH_t, nW_t);
    	convert1Dto2D(pTargetImage, target_img_Y, target_img_U, target_img_V, nH, nW);

    	Tensor3D* target = new Tensor3D(nH, nW, 1);
    	for (int h = 0; h < nH; h++)
        	for (int w = 0; w < nW; w++)
            	target->set_elem(h, w, 0, target_img_Y[h][w]);
		
		// 학습 루프
    	for (int e = 0; e < epochs; e++) {
        	cout << "[Epoch " << e+1 << "/" << epochs << "]" << endl;
        	train_step(tensors[0], target, lr);
    	}

		// 메모리 해제
    	free(pLoadImage);
    	free(pTargetImage);
    	free_dmatrix2D(input_img_Y, nH, nW);
    	free_dmatrix2D(input_img_U, nH, nW);
    	free_dmatrix2D(input_img_V, nH, nW);
    	free_dmatrix2D(target_img_Y, nH, nW);
    	free_dmatrix2D(target_img_U, nH, nW);
    	free_dmatrix2D(target_img_V, nH, nW);
    	delete target;


	}

	void train_step(const Tensor3D* input, const Tensor3D* target, double lr) {
		// (구현할 것)
		// 동작: 이미 로드된 (input, target) 한 쌍으로 1회의 학습 스텝을 수행
		// 아래 타이밍 구조에 맞춰 (a)~(e)를 구현할 것

		double loss_val = 0.0; // (b)에서 계산된 loss 값을 이 변수에 저장할 것

		double t0 = omp_get_wtime();

		//(a) forward pass
		// (구현할 것): (a) forward pass
		// tensors[i] -> layers[i]->forward() -> tensors[i+1] 순서로 수행
		while (tensors.size() > 1) { delete tensors.back(); tensors.pop_back(); }
    	for (int i = 0; i < (int)layers.size(); i++) {
        	Tensor3D* output = layers[i]->forward(tensors[i]);
        	tensors.push_back(output);
    	}

		double t1 = omp_get_wtime();

		// (b) loss 계산
		// (구현할 것): (b) loss 계산 - MSELoss를 사용하여 pred와 target의 loss 계산
		//                              결과를 loss_val에 저장할 것
		Tensor3D* pred = tensors.back();
    	MSELoss loss_fn;
    	loss_val = loss_fn.compute(pred, target);

		// (c) backward pass
		// (구현할 것): (c) backward pass
		// 힌트:
		// Tensor3D* grad = loss_fn.gradient(pred, target);
		// for (int i = (int)layers.size()-1; i >= 0; i--) {
		//     Tensor3D* grad_prev = layers[i]->backward(grad);
		//     delete grad;
		//     grad = grad_prev;
		// }
		// delete grad;

		Tensor3D* grad = loss_fn.gradient(pred, target);
    	for (int i = (int)layers.size()-1; i >= 0; i--) {
        	Tensor3D* grad_prev = layers[i]->backward(grad);
        	delete grad;
        	grad = grad_prev;
    	}
    	delete grad;

		double t2 = omp_get_wtime();

		// (d) weight 업데이트
		// (구현할 것): (d) weight 업데이트 - 모든 layer의 update_weights(lr) 호출
		
		for (auto l : layers) l->update_weights(lr);

		double t3 = omp_get_wtime();

		// (e) 중간/출력 tensor 삭제 (tensors[0]만 유지)
		// (구현할 것): (e) 중간/출력 tensor 삭제 (입력 tensor인 tensors[0]만 유지)

		while (tensors.size() > 1) { delete tensors.back(); tensors.pop_back(); }

		cout << "  Loss: " << loss_val
		     << " | Forward: "  << t1 - t0 << "s"
		     << " | Backward: " << t2 - t1 << "s"
		     << " | Update: "   << t3 - t2 << "s" << endl;
	}

	// =========================================================================
	// helper / utility
	// =========================================================================
	void read_image(const string filename, byte*& pLoadImage, double**& img_Y, double**& img_U, double**& img_V, int& nH, int& nW) {
		LoadBmp(filename.c_str(), &pLoadImage, nH, nW);

		img_Y = dmatrix2D(nH, nW);
		img_U = dmatrix2D(nH, nW);
		img_V = dmatrix2D(nH, nW);
		convert1Dto2D(pLoadImage, img_Y, img_U, img_V, nH, nW);

		Tensor3D* temp = new Tensor3D(nH, nW, 1);
		for (int h = 0; h < nH; h++)
			for (int w = 0; w < nW; w++)
				temp->set_elem(h, w, 0, img_Y[h][w]);
		tensors.push_back(temp);
	}
	void save_image(string filename, byte*& pLoadImage, Tensor3D*& tensor_Y, double** img_U, double** img_V, int nH, int nW) {
		double** img_Y = dmatrix2D(nH, nW);
		for (int h = 0; h < nH; h++)
			for (int w = 0; w < nW; w++)
				img_Y[h][w] = tensor_Y->get_elem(h, w, 0);
		convert2Dto1D(img_Y, img_U, img_V, pLoadImage, nH, nW);
		SaveBmp(filename.c_str(), pLoadImage, nH, nW);
		free_dmatrix2D(img_Y, nH, nW);
	}
	void print_layer_info() const {
		cout << endl << "(Layer information)_____________" << endl;
		for (unsigned i = 0; i < layers.size(); i++) {
			cout << i + 1 << "-th layer: ";
			layers.at(i)->print();
		}
	}
	void print_tensor_info() const {
		cout << endl << "(Tensor information)_____________" << endl;
		for (unsigned i = 0; i < tensors.size(); i++) {
			cout << i + 1 << "-th tensor: ";
			tensors.at(i)->print();
		}
	}
};
