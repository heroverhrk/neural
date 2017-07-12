#include "def.h"

MatrixXd sigmoid(MatrixXd x){
  x = x.array() * -1.0;
  x = x.array().exp();
  x = x.array() + 1.0;
  x = x.array().inverse()*1.0;

  return x;
}

MatrixXd deriv_sigmoid(MatrixXd x){
  MatrixXd sig1 = sigmoid(x);
  MatrixXd sig2 = sig1.array() * -1.0 + 1.0;
  x = sig1.array() * sig2.array();
  return x;
}

MatrixXd ReLU(MatrixXd x) {
  int rows = x.rows();
  int cols = x.cols();
  MatrixXd zero = MatrixXd::Zero(rows,cols);
  x = x.array().max(zero.array());
  return x;
}

MatrixXd deriv_ReLU(MatrixXd x){
  x = ReLU(x);
  x = x.array()/x.array();
  x = ReLU(x);
  return x;
}

MatrixXd softmax(MatrixXd x){
  x = x.array().exp();
  double xx = x.sum();
  x = x.array() / xx;
  return x;
}
