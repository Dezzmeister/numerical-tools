#include "numerical.h"
#include <stdlib.h>
#include <stdio.h>

std::string stringify(IterativeResult result) {
    std::string out = "x: " + std::to_string(result.xf) 
                    + ", n: " + std::to_string(result.n) 
                    + ", converged: " + (result.converged ? "T" : "F")
                    + ((result.periodicity == -1) ? "" 
                            : (", periodicity: " + std::to_string(result.periodicity)));
    return out;
}

bool approx_equals(double a, double b) {
    return fabs(a - b) < EPSILON;
}

double deriv(StateFunc f, double x, double params[]) {
    double x1 = x + EPSILON;
    double x0 = x - EPSILON;

    return (f(x1, params) - f(x0, params)) / (x1 - x0);
}

IterativeResult basic_iterative(int n, StateFunc g, double x0, double params[], double * xs, int max_period, double * cycle) {
    double x = x0;
    double xp = HUGE_VAL;
    
    for (int i = 0; i < n; i++) {
        xp = x;
        x = g(x, params);
        
        if (xs != nullptr) {
            xs[i] = x;
        }
    }

    IterativeResult result;
    result.xf = x;
    result.n = n;
    result.converged = (fabs(x - xp) < EPSILON);
    result.periodicity = -1;

    if (max_period > 0) {
        double xi = x;

        for (int i = 0; i < max_period; i++) {
            x = g(x, params);

            if (cycle != nullptr) {
                cycle[i] = x;
            }
           
            if (approx_equals(x, xi)) {
                result.periodicity = i + 1;
                return result;
            }
        }
    }

    return result;
}

IterativeResult newton_iterative(int n, StateFunc f, double x0, double params[], double * xs, int max_period, double * cycle) {
    // Compose f and f' into new state function g
    StateFunc g = [f](double x, double params[]) {
        return x - (f(x, params) / deriv(f, x, params));
    };
    
    return basic_iterative(n, g, x0, params, xs, max_period, cycle);
}

double find_bifurcation_right(IterativeResultGenerator generator, double r0, double init_stepsize, double precision) {
    double r = r0;
    double stepsize = init_stepsize;
    int cycles = -1;
    int next_cycles;
    IterativeResult res;

    res = generator(r0);
    cycles = res.periodicity;

    while (stepsize >= precision) {
        res = generator(r + stepsize);
        next_cycles = res.periodicity;

        if (next_cycles != cycles) {
            stepsize /= 2;
        } else {
            r += stepsize;
        }
    }

    return r;
}

double find_change_right(IterativeResultGenerator generator, double r0, double init_stepsize) {
    double r = r0;
    double stepsize = init_stepsize;
    double xp = HUGE_VAL;
    double xn;
    IterativeResult res;

    res = generator(r0);
    xp = res.xf;

    while (stepsize >= (EPSILON / 2.0)) {
        res = generator(r + stepsize);
        xn = res.xf;

        if (!approx_equals(xp, xn)) {
            stepsize /= 2;
        } else {
            r += stepsize;
        }
    }

    return r;
}

