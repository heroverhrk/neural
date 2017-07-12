#include "def.h"

using namespace std;
using namespace Eigen;

const string train_data = "mnist/train-images-idx3-ubyte";
const string train_label = "mnist/train-labels-idx1-ubyte";
const string test_data = "mnist/t10k-images-idx3-ubyte";
const string test_label = "mnist/t10k-labels-idx1-ubyte";

namespace mnist {
    // big endian -> little endian
    int convert_endian(unsigned char* b){
      int hoge = 0;
      for(int i=0; i<4; i++){
        hoge = (hoge<<8) | b[i];
      }
      return hoge;
    }

    //load data
    MatrixXd load_data(const string& file){
      ifstream in;
      in.open(file.c_str(), ios::in | ios::binary);

      if(!in){
        cout << "cannot open file: " << file << endl;
        exit(1);
      }
      //magic number
      unsigned char b[4];
      in.read((char*) b, sizeof(char) * 4);
      //data size
      in.read((char*) b, sizeof(char) * 4);
      int num = convert_endian(b);
      //data height(row)
      in.read((char*) b, sizeof(char) * 4);
      const int height = convert_endian(b);
      //data width(col)
      in.read((char*) b, sizeof(char) * 4);
      const int width = convert_endian(b);

      //pixel data
      unsigned char* hogehoge = new unsigned char[height * width];
      MatrixXd hoge(height*width, num);
      for(int j=0; j<num; j++){
        in.read((char*) hogehoge, sizeof(char) * height * width);
        for(int i=0; i<height*width; i++){
          hoge(i,j) = hogehoge[i]/256.0;
        }
      }
      delete[] hogehoge;
      in.close();
      cout << "success load file: " << file << endl;
      return hoge;
    }

    //load label
    MatrixXd load_label(const string& file){
      ifstream in;
      in.open(file.c_str(), ios::in | ios::binary);
      if(!in){
        cout << "cannot load file: " << file <<endl;
        exit(1);
      }

      //magic number
      unsigned char b[4];
      in.read((char*) b, sizeof(char) * 4);
      //data size
      in.read((char*) b, sizeof(char) * 4);
      int num = convert_endian(b);

      //label data
      MatrixXd hoge(10, num);
      for(int i=0; i < num; i++){
        char digit;
        in.read((char*) &digit, sizeof(char));
        hoge(digit, i) = 1.0;
      }
      in.close();
      cout << "success load file: " << file << endl;
      return hoge;
    }

      MatrixXd load_train_data(){
        return load_data(train_data);
      }
      MatrixXd load_train_label(){
        return load_label(train_label);
      }
      MatrixXd load_test_data(){
        return load_data(test_data);
      }
      MatrixXd load_test_label(){
        return load_label(test_label);
      }
}
