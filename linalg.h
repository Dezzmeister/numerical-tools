#ifndef LINALG_H
#define LINALG_H

#include <functional>

#define ZERO_IN_DIAGONAL_ERROR      10
#define BACKSUB_ZERO_ERROR          20
#define SIZE_ERROR                  30

typedef std::function<double(double, double)> Map2;

const Map2 ADD = [](double a, double b) {
    return a + b;
};

const Map2 SUBTRACT = [](double a, double b) {
    return a - b;
};

const Map2 MULTIPLY = [](double a, double b) {
    return a * b;
};

const Map2 DIVIDE = [](double a, double b) {
    return a / b;
};

class Vector {
    public:
        double * values;
        const unsigned int N;

        Vector(unsigned int _N);
        Vector(unsigned int _N, double * _values);
        
        Vector(const Vector& v);
        ~Vector();

        double& operator[](unsigned int n) const;
        double& operator()(unsigned int n) const;
        const Vector operator+(const Vector& v) const;
        const Vector operator-(const Vector& v) const;
        
        const Vector dot(const Vector& v) const;
        const Vector map2(const Vector& v, Map2 map) const;
        const double sum() const;
        const void print() const;

        static Vector one_hot(unsigned int N, unsigned int i);
};

class Matrix {
    public:
        double ** values;
        const unsigned int N;

        Matrix(unsigned int _N);
        Matrix(unsigned int _N, double ** _values);
        
        template <unsigned int n> Matrix(double _values[n][n]): N(n) {
            malloc_matrix();

            for (unsigned int i = 0; i < N; i++) {
                for (unsigned int j = 0; j < N; j++) {
                    values[i][j] = _values[i][j];
                }
            }
        }

        Matrix(const Matrix& m);
        ~Matrix();

        double& operator()(unsigned int row, unsigned int col) const;
        const Matrix operator+(const Matrix& m) const;
        const Matrix operator-(const Matrix& m) const;
        const Matrix& operator+=(const Matrix& m);
        const Matrix operator*(const Matrix& m) const;
        const Vector operator*(const Vector& v) const;
        const Vector operator/(const Vector& b) const;

        const Matrix map2(const Matrix& m, Map2 map) const;
        const void print() const;
        void gauss(Vector& b);
        const Vector back_sub(const Vector& b) const;
        const Vector forward_sub(const Vector& b) const;
        const void lu_decomp(Matrix& L, Matrix& U) const;
        void set_diagonal(double d);
        void set_column(unsigned int col, Vector& v);
        void set_row(unsigned int row, Vector& v);
        const Vector solve(const Vector& b) const;
        Matrix inverse();
        double det();

    private:
        void malloc_matrix();
};

#endif
