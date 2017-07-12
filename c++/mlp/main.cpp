#include "def.h"
#include "mnist.h"
#include "layer.h"

int main(){
  srand( (unsigned)time(NULL) );

  MatrixXd train_data = mnist::load_train_data();
  MatrixXd train_label = mnist::load_train_label();

  vector<layer::dense*> layers(2);
  layers[0] = new layer::dense(784, 100, ReLU, deriv_ReLU);
  layers[1] = new layer::dense(100, 10, softmax, ReLU);

  layers[0]->feed(train_data.col(0));
  layers[1]->feed(layers[0]->y);
  layers[1]->backprop_out(train_label.col(0));
  layers[0]->backprop(*layers[1]);

  cout << layers[1]->y << endl;

  return 0;
}
