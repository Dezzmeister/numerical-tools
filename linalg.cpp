#include "linalg.h"

#include <string.h>

Vector::Vector(unsigned int _N): N(_N) {
    values = (double *) calloc(N, sizeof(double));
}

Vector::Vector(unsigned int _N, double * _values): N(_N) {
    values = (double *) calloc(N, sizeof(double));
    memcpy(values, _values, sizeof(double) * N);
}

Vector::Vector(const Vector& v): N(v.N) {
    values = (double *) calloc(N, sizeof(double));
    memcpy(values, v.values, sizeof(double) * N);
}

Vector::~Vector() {
    free(values);
}

double& Vector::operator[](unsigned int n) const {
    return values[n];
}

double& Vector::operator()(unsigned int n) const {
    return values[n];
}

const Vector Vector::operator+(const Vector& v) const {
    return map2(v, ADD);
}

const Vector Vector::operator-(const Vector& v) const {
    return map2(v, SUBTRACT);
}

const Vector Vector::dot(const Vector& v) const {
    return map2(v, MULTIPLY).sum();
}

const Vector Vector::map2(const Vector& v, Map2 map) const {
    Vector out(N);

    for (unsigned int i = 0; i < N; i++) {
        out[i] = map(values[i], v[i]);
    }

    return out;
}

const double Vector::sum() const {
    double out = 0;

    for (unsigned int i = 0; i < N; i++) {
        out += values[i];
    }

    return out;
}

const void Vector::print() const {
    unsigned int i;

    printf("[");
    for (i = 0; i < N; i++) {
        printf("%.3f%s", values[i], (i == (N - 1)) ? "]\n" : "\t");
    }
}

Vector Vector::one_hot(unsigned int N, unsigned int i) {
    Vector out(N);

    out[i] = 1;

    return out;
}

Matrix::Matrix(unsigned int _N): N(_N) {
    malloc_matrix();
}

Matrix::Matrix(unsigned int _N, double ** _values): N(_N) {
    malloc_matrix();
    memcpy(values, _values, sizeof(double *) * N);
        
    for (unsigned int i = 0; i < N; i++) {
        memcpy(values[i], _values[i], sizeof(double) * N);
    }
}

Matrix::Matrix(const Matrix& m): N(m.N) {
    malloc_matrix();

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            values[i][j] = m.values[i][j];
        }
    }
}

Matrix::~Matrix() {
    for (unsigned int i = 0; i < N; i++) {
        free(values[i]);
    }

    free(values);
}

double& Matrix::operator()(unsigned int row, unsigned int col) const {
    return values[row][col];
}

const Matrix Matrix::operator+(const Matrix& m) const {
    return map2(m, ADD);
}

const Matrix Matrix::operator-(const Matrix& m) const {
    return map2(m, SUBTRACT);
}

const Matrix& Matrix::operator+=(const Matrix& m) {
    if (m.N != N) {
        throw SIZE_ERROR;
    }

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            values[i][j] += m(i, j);
        }
    }

    return *this;
}

const Vector Matrix::operator*(const Vector& v) const {
    if (v.N != N) {
        throw SIZE_ERROR; 
    }

    Vector out(N);

    for (unsigned int i = 0; i < N; i++) {
        double sum = 0;

        for (unsigned int j = 0; j < N; j++) {
            sum += values[i][j] * v[j];
        }

        out[i] = sum;
    }

    return out;
}

const Matrix Matrix::operator*(const Matrix& m) const {
    if (m.N != N) {
        throw SIZE_ERROR;
    }

    Matrix out(N);

    for (unsigned int i = 0; i < N; i++) {            
        for (unsigned int j = 0; j < N; j++) {
            double sum = 0;
                
            for (unsigned int k = 0; k < N; k++) {
                sum += (values[i][k] * m(k, j));
            }

            out(i, j) = sum;
        }
    }

    return out;
}

const Vector Matrix::operator/(const Vector& b) const {
    return solve(b);
}

const Matrix Matrix::map2(const Matrix& m, Map2 map) const {
    if (m.N != N) {
        throw SIZE_ERROR;
    }

    Matrix out(N);

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            out(i, j) = map(values[i][j], m(i, j));       
        }
    }

    return out;
}

const void Matrix::print() const {
    unsigned int i, j;

    printf("[");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++) {
            printf("%.3f%s", values[i][j], (j == (N - 1)) ? "" : "\t");
        }
        printf("]%s", (i == (N - 1)) ? "]\n" : "\n");
    }
}

void Matrix::gauss(Vector& b) {
    unsigned int i, k, j;
    double M;
        
    for (k = 0; k < N - 1; k++) {
        if (values[k][k] == 0.0) {
            throw ZERO_IN_DIAGONAL_ERROR;
        }

        for (i = k + 1; i < N; i++) {
            M = values[i][k]/values[k][k];
            b[i] -= M * b[k];

            for (j = 0; j < N; j++) {
                values[i][j] -= M * values[k][j];
            }
        }
    }
}

const Vector Matrix::back_sub(const Vector& b) const {
    Vector x(N);
    int k;
    unsigned int j;

    if (values[N - 1][N - 1] == 0.0) {
        throw BACKSUB_ZERO_ERROR;
    }

    x[N - 1] = b[N - 1]/values[N - 1][N - 1];
    for (k = N - 2; k >= 0; k--) {
        x[k] = b[k];

        for (j = k + 1; j < N; j++) {
            x[k] -= values[k][j] * x[j];
        }

        x[k] /= values[k][k];
    }

    return x;
}

// https://www.researchgate.net/publication/220624910_Energy_Efficient_Hardware_Architecture_of_LU_Triangularization_for_MIMO_Receiver
const void Matrix::lu_decomp(Matrix& L, Matrix& U) const {
    Matrix B(*this);
    unsigned int i, j, k;

    for (k = 0; k < N; k++) {
        U(k, k) = B(k, k);
            
        for (i = k + 1; i < N; i++) {
            L(i, k) = B(i, k)/U(k, k);
            U(k, i) = B(k, i);
        }

        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                B(i, j) -= (L(i, k) * U(k, j));
            }
        }
    }

    L.set_diagonal(1);
}
    
const Vector Matrix::forward_sub(const Vector& b) const {
    unsigned int k, j;
    Vector x(N);

    x[0] = b[0];

    for (k = 1; k < N; k++) {
        x[k] = b[k];
            
        for (j = 0; j < k; j++) {
            x[k] = x[k] - (values[k][j] * x[j]);
        }
    }

    return x;
}

void Matrix::set_diagonal(double d) {
    for (unsigned int i = 0; i < N; i++) {
        values[i][i] = d;
    }
}

void Matrix::set_column(unsigned int col, Vector& v) {
    if (v.N != N) {
        throw SIZE_ERROR;
    }

    for (unsigned int i = 0; i < N; i++) {
        values[i][col] = v[i];
    }
}

void Matrix::set_row(unsigned int row, Vector& v) {
    if (v.N != N) {
        throw SIZE_ERROR;
    }

    for (unsigned int i = 0; i < N; i++) {
        values[row][i] = v[i];
    }
}

const Vector Matrix::solve(const Vector& b) const {
    Matrix L(N);
    Matrix U(N);
    lu_decomp(L, U);
    Vector y = L.forward_sub(b);
    Vector x = U.back_sub(y);

    return x;
}
    
Matrix Matrix::inverse() {
    Matrix out(N);

    for (unsigned int i = 0; i < N; i++) {
        Vector b = Vector::one_hot(N, i);
        Vector x = solve(b);
        out.set_column(i, x);
    }

    return out;
}

double Matrix::det() {
    Matrix L(N);
    Matrix U(N);
    lu_decomp(L, U);

    double prod = 1;

    for (unsigned int i = 0; i < N; i++) {
        prod *= U(i, i);
    }

    return prod;
}

void Matrix::malloc_matrix() { 
    values = (double **) calloc(N, sizeof(double *));

    for (unsigned int i = 0; i < N; i++) {
        values[i] = (double *) calloc(N, sizeof(double));
    }
}
