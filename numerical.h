#ifndef NUMERICAL_H
#define NUMERICAL_H

#include <functional>
#include <string>
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-6;
const double PI = 3.14159265358979323846;
const double TWO_PI = 2 * PI;

/**
 * A function that takes the current state (x_n) and produces the next state (x_{n+1})
 *
 * g(double x, double params[])
 *      x: the variable
 *      params: array of constant parameters
 */
typedef std::function<double(double, double[])> StateFunc;

typedef struct {
    /* Final x */
    double xf;

    /* Number of iterations */
    int n;

    /* Did the sequence converge (consecutive elements of the sequence equal within EPSILON) */
    bool converged;

    /** 
     * Period of iterative result. -1 indicates that this was not set or that
     * the number of elements in the cycle exceeded 'max_period'
     */
    int periodicity;
} IterativeResult;

typedef std::function<IterativeResult(double)> IterativeResultGenerator;

std::string stringify(IterativeResult result);

/**
 * Returns true if two doubles are equal within EPSILON.
 */
bool approx_equals(double a, double b);

/**
 * Numerical derivative of f at a point (x, f(x))
 */
double deriv(StateFunc f, double x, double params[]);

/**
 * Applies the basic iterative method to g(x) starting with x = g(x0). Returns a struct with some results.
 *
 * n: number of iterations
 * g: state function
 * x0: initial x
 * params: constant parameters of g
 * xs: optional array of size n to hold each x in the sequence x_{n+1} = g(x_n). Does not include x0
 * max_period: -1 by default, set this to a positive integer to look for cycles in the sequence.
 *      Will only look for a cycle of up to 'max_period' elements; will not look for a cycle if set to -1
 * cycle: optional array of size 'max_period' to hold elements of a cycle. Each entry in the cycle will appear
 *      in this array only once
 */
IterativeResult basic_iterative(int n, StateFunc g, double x0, double params[], double * xs = nullptr, int max_period = -1, double * cycle = nullptr);

IterativeResult newton_iterative(int n, StateFunc f, double x0, double params[], double * xs = nullptr, int max_period = -1, double * cycle = nullptr);

/**
 * Finds a bifurcation (a fixed point of some g(x) for which the number of cycles increases). Given a wrapper
 * around the function g(x) with parameter r, an initial r (less than the r at which the bifurcation occurs),
 * and an initial stepsize, this function steps r forward and decreases the stepsize appropriately
 * until finding the r at which the bifurcation occurs, within EPSILON.
 *
 * generator: A function only of r, that gives an IterativeResult. This function may call
 *      `basic_iterative` with r as a parameter to the state function being analyzed.
 * r0: An initial guess for the desired r. r0 must be less than r, because this function will
 *      increase r until finding the point at which the first bifurcation occurs after r0.
 *      For example, if trying to find the point at which a period-2 cycle bceomes a period-4 cycle,
 *      r0 should be from the set of rs that gives the period-2 cycle.
 * init_stepsize: Initial stepsize. The choice of initial stepsize is not so important, but an optimal
 *      choice should overshoot the bifurcation when added to r0.
 * precision: Maximum width of the stepsize for which the algorithm can successfully return.
 *      The algorithm will return an r, +/- this precision
 */
double find_bifurcation_right(IterativeResultGenerator generator, double r0, double init_stepsize, double precision = (EPSILON / 2.0));

/**
 * Much like `find_bifurcation_right`, except this function looks for a change in the limiting xf
 * instead of a bifurcation.
 */
double find_change_right(IterativeResultGenerator generator, double r0, double init_stepsize);

#endif
