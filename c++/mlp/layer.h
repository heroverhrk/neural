#include "def.h"
#include "activation.h"

using namespace std;
using namespace Eigen;

namespace layer{
  class dense{
    int in_dim;
    int out_dim;
    function<MatrixXd(MatrixXd)> feed_func;
    function<MatrixXd(MatrixXd)> back_func;
    MatrixXd x;
    MatrixXd u;
    MatrixXd delta;
    float eps = 0.01;

  public:
    MatrixXd y;
    MatrixXd w;
    MatrixXd b;
    MatrixXd dw;
    MatrixXd db;

    dense(int in, int out, std::function<MatrixXd(MatrixXd)> ffunc, std::function<MatrixXd(MatrixXd)> bfunc){
      in_dim = in;
      out_dim = out;
      feed_func = ffunc;
      back_func = bfunc;
      w = MatrixXd::Random(in_dim,out_dim);
      b = MatrixXd::Random(out_dim,1);
    }
    void feed(MatrixXd input){
      x = input;
      u = w.transpose() * x + b;
      y = feed_func(u);
    }
    void backprop_out(MatrixXd t){
      delta = y - t;
      dw = x * delta.transpose();
      db = delta;

      w = w.array() - dw.array() * eps;
      b = b.array() - db.array() * eps;
    }
    void backprop(dense den){
      delta = back_func(u).array() * (den.w * den.delta).array();
      dw = x * delta.transpose();
      db = delta;

      w = w.array() - dw.array() * eps;
      b = b.array() - db.array() * eps;
    }
  };
}
