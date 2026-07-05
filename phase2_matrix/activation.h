#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <memory>

// Abstract base class for all activation functions
class Activation
{
    public:
        // Apply activation to input vector (element-wise)
        virtual double apply(double x) const = 0;

        // Compute derivative at a point (for backprop)
        virtual double derivative(double x) const = 0;

        virtual ~Activation() = default;
};

// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
class Sigmoid : public Activation
{
    public:
        double apply(double x) const override
        {
            return 1.0 / (1.0 + exp(-x));
            // σ(x) = 1 / (1 + e^(-x))
        }

        // Derivative using the output: f'(x) = f(x) * (1 - f(x))
        double derivative(double x) const override
        {
            // derivative of the sigmoid is: σ'(x) = σ(x) × (1 - σ(x))
            double s = apply(x);
            return s * (1.0 - s);
            //So if s = sigmoid(x), then derivative = s * (1 - s). That's why the code does apply(x) first to get s, then computes s * (1 - s).
        }
};

// ReLU activation: f(x) = max(0, x)
class ReLu : public Activation
{
    public:
        double apply(double x) const override
        {
            return x > 0 ? x : 0;
        }

        // Derivative: 1 if x > 0, else 0
        double derivative(double x) const override
        {
            return x > 0 ? 1.0 : 0.0;
        }
};

#endif