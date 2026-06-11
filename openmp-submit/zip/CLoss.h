#pragma once
#include "CTensor.h"

// Loss 함수 기반 클래스
class Loss {
public:
    virtual ~Loss() {}
    virtual double compute(const Tensor3D* pred, const Tensor3D* target) = 0;
    virtual Tensor3D* gradient(const Tensor3D* pred, const Tensor3D* target) = 0;
};

// MSE Loss: L = (1/N) * sum((pred - target)^2)
class MSELoss : public Loss {
public:
    double compute(const Tensor3D* pred, const Tensor3D* target) override {
        int nH, nW, nC;
        pred->get_info(nH, nW, nC);
        double loss = 0.0;
        int total = nH * nW * nC;
        for (int h = 0; h < nH; h++)
            for (int w = 0; w < nW; w++)
                for (int c = 0; c < nC; c++) {
                    double diff = pred->get_elem(h, w, c) - target->get_elem(h, w, c);
                    loss += diff * diff;
                }
        return loss / total;
    }

    // dL/dpred = 2*(pred - target) / N
    Tensor3D* gradient(const Tensor3D* pred, const Tensor3D* target) override {
        int nH, nW, nC;
        pred->get_info(nH, nW, nC);
        int total = nH * nW * nC;
        Tensor3D* grad = new Tensor3D(nH, nW, nC);
        for (int h = 0; h < nH; h++)
            for (int w = 0; w < nW; w++)
                for (int c = 0; c < nC; c++) {
                    double g = 2.0 * (pred->get_elem(h, w, c) - target->get_elem(h, w, c)) / total;
                    grad->set_elem(h, w, c, g);
                }
        return grad;
    }
};
