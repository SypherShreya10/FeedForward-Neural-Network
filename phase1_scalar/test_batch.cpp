#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "activation.h"
#include "layer_matrix.h"

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXd X (4, 2);
    X << 0.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         1.0, 1.0;

    cout << "Input batch X (4 samples, 2 features each): " << endl << X << endl;

    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    //creating layers
    DenseLayerMatrix layer1 (2, 4, sigmoid);
    DenseLayerMatrix layer2 (4, 1, sigmoid);

    //forward pass
    MatrixXd hidden = layer1.forward(X);
    cout << "Hidden matrix: " << endl << hidden << endl;
    MatrixXd output = layer2.forward(hidden);
    cout << "Output: " << endl << output << endl;

    return 0;

}