#pragma once
#include "Imagelib.h"

// Tensor3D는 크기가 (nH x nW x nC)인 3차원 tensor를 관리함

class Tensor3D {
private:
	double* tensor;  // 연속 메모리 배열: 인덱스 = h * nW * nC + w * nC + c
	int nH; // height
	int nW; // width
	int nC; // channel
public:
	Tensor3D(int _nH, int _nW, int _nC) : nH(_nH), nW(_nW), nC(_nC) {
		// (구현할 것)
		// 동작: new double[nH * nW * nC]()로 연속 메모리를 할당하여 tensor에 저장
		//       끝의 ()는 모든 element를 0으로 초기화함
	}
	~Tensor3D() {
		// (구현할 것)
		// 동작: delete[] tensor 로 메모리 해제
	}
	void set_elem(int _h, int _w, int _c, double _val) { tensor[_h * nW * nC + _w * nC + _c] = _val; }
	double get_elem(int _h, int _w, int _c) const {
		// (구현할 것)
		// 동작: 행=_h, 열=_w, 채널=_c 위치 element를 반환할 것
		// 인덱스 계산: _h * nW * nC + _w * nC + _c
	}

	void get_info(int& _nH, int& _nW, int& _nC) const {
		// (구현할 것)
		// 동작: 행렬의 차원(nH, nW, nC)을 pass by reference로 반환
	}

	void print() const {
		// (구현할 것)
		// 동작: 행렬의 크기 (nH x nW x nC)를 화면에 출력
	}
};
