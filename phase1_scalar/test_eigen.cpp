#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    // Create a simple 2x3 matrix
    MatrixXd m(2, 3);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 4;
    m(1, 1) = 5;
    m(1, 2) = 6;

    cout << "Matrix m: " << endl << m << endl;
    
    return 0;
}