#include "ImageLib.h"
#include "CModel.h"
using namespace std;

// 정확하게 동작시 20점 (부분점수 없음)

// 모델 구조를 생성하는 헬퍼 함수 (매 측정마다 동일한 초기 가중치에서 시작하기 위해 사용)
Model* build_model() {
	Model* model = new Model();
	model->add_layer(new Layer_Conv("Conv1", 9, 1, 64, LOAD_INIT, "model/weights_conv1_9x9x1x64.txt", "model/biases_conv1_64.txt"));
	model->add_layer(new Layer_ReLU("Relu1", 1, 64, 64));
	model->add_layer(new Layer_Conv("Conv2", 5, 64, 32, LOAD_INIT, "model/weights_conv2_5x5x64x32.txt", "model/biases_conv2_32.txt"));
	model->add_layer(new Layer_ReLU("Relu2", 1, 32, 32));
	model->add_layer(new Layer_Conv("Conv3", 5, 32, 1, LOAD_INIT, "model/weights_conv3_5x5x32x1.txt", "model/biases_conv3_1.txt"));
	return model;
}

int main() {

	// =========================================================================
	// OpenMP 성능 측정 (2 epoch 기준, 매번 동일한 초기 가중치에서 시작)
	// =========================================================================
	cout << "(OpenMP Performance)_____________" << endl;
	double t_serial = 0.0;

	// --- 1 thread (serial baseline) ---
	{
		omp_set_num_threads(1);

		Model* model = build_model();
		double t_start = omp_get_wtime();
		model->train("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp", 0.0001, 1);
		t_serial = omp_get_wtime() - t_start;
		cout << "Threads:  1 | Time: " << t_serial << "s | Speedup: 1.00x" << endl;
		delete model;
	}

	// --- 2 threads ---
	{
		omp_set_num_threads(2);

		Model* model = build_model();
		double t_start = omp_get_wtime();
		model->train("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp", 0.0001, 1);
		double t_n = omp_get_wtime() - t_start;
		cout << "Threads:  2 | Time: " << t_n << "s | Speedup: " << t_serial / t_n << "x" << endl;
		delete model;
	}

	// --- 4 threads ---
	{
		omp_set_num_threads(4);

		Model* model = build_model();
		double t_start = omp_get_wtime();
		model->train("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp", 0.0001, 1);
		double t_n = omp_get_wtime() - t_start;
		cout << "Threads:  4 | Time: " << t_n << "s | Speedup: " << t_serial / t_n << "x" << endl;
		model->test("baby_512x512_input.bmp", "baby_512x512_output_backprop.bmp");
		model->print_layer_info();
		model->print_tensor_info();
		delete model;
	}

	system("PAUSE");
	return 0;
}
