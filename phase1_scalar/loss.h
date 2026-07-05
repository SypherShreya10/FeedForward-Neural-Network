#ifndef LOSS_H
#define LOSS_H

#include <cmath>

// Abstract base class for loss functions
class Loss
{
    public:
        // Compute loss value
        virtual double value(double pred, double target) const = 0;

        // Compute gradient of loss w.r.t. prediction
        virtual double gradient(double pred, double target) const = 0;

        virtual ~Loss() = default;
};

// Mean Squared Error: L = 0.5 * (pred - target)^2
class MSE : public Loss
{
    public:
        double value(double pred, double target) const override
        {
            double diff = pred - target;
            return 0.5 * diff * diff;
        }

        double gradient(double pred, double target) const override
        {
            return pred - target;
        }
};

#endif