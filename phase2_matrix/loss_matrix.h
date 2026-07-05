#ifndef LOSS_MATRIX_H
#define LOSS_MATRIX_H

#include <Eigen/Dense>
#include <memory>

using namespace std;
using namespace Eigen;

class LossMatrix
{
    public:
        // Compute loss for a batch
        // predictions: (batch_size x output_size)
        // targets: (batch_size x output_size)
        // Returns: scalar loss value (average across batch)
        virtual double compute(const MatrixXd& predictions, const MatrixXd& targets) const = 0;
        
        // Compute gradient for a batch
        // Returns: (batch_size x output_size) matrix of gradients
        virtual MatrixXd gradient(const MatrixXd& predictions, const MatrixXd& targets) const = 0;
        
        virtual ~LossMatrix() = default;
};

class MSEMatrix : public LossMatrix
{
    public:
        double compute(const MatrixXd& predictions, const MatrixXd& targets) const override
        {
            int batch_size = predictions.rows();
            int output_size = predictions.cols();
            double total_loss = 0.0;

            for (int i=0; i<batch_size; i++)
            {
                for (int j=0; j<output_size; j++)
                {
                    total_loss += ((predictions(i, j) - targets(i, j)) * (predictions(i, j) - targets(i, j)));
                }
            }

            return total_loss / (batch_size * output_size);
        }

        MatrixXd gradient(const MatrixXd& predictions, const MatrixXd& targets) const override
        {
            int batch_size = predictions.rows();
            int output_size = predictions.cols();

            MatrixXd grad(batch_size, output_size);

            for (int i=0; i<batch_size; i++)
            {
                for (int j=0; j<output_size; j++)
                {
                    grad(i, j) = predictions(i, j) - targets(i, j);
                }
            }

            return grad;
        }
};

#endif