////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     inc/matrix.h
// CREATED:  2016-04-26 17:31:45
// MODIFIED: 2016-04-26 18:09:09
////////////////////////////////////////////////////////////////////
#ifndef INC/MATRIX_H
    #define INC/MATRIX_H
    #include <stddef.h>

    struct _matrix * matrix;

    /* Creates a rows x cols matrix with all values 0. */
    matrix newMatrix(int rows, int cols);

    /* Copies a matrix. */
    matrix copyMatrix(matrix mtx);

    /* Deletes a matrix. */
    void deleteMatrix(matrix mtx);

    /* Sets the (row, col) element of mtx to val.  Returns 0 if
     * successful, and -1 if row or col are outside of the 
     * dimensions of mtx.
     */
    int setElement(matrix mtx, int row, int col, double val);

    /* Sets the reference val to value of the (row, col) 
     * element of mtx.  Returns 0 if successful, -1 if val is
     * NULL, and -2 if row or col are outside of the dimensions 
     * of mtx.
     */
    int getElement(matrix mtx, int row, int col, double * val);

    int nRows(matrix mtx);

    int nCols(matrix mtx);

    void printMatrix(matrix mtx);

    /* Writes the transpose of matrix in into matrix out.  
     * Returns 0 if successful, and -1 if the dimensions of in 
     * and out are incompatible.
     */
    int transpose(matrix in, matrix out);

    /* Writes the sum of matrices mtx1 and mtx2 into matrix sum.
     * Returns 0 if successful, and -1 if the dimensions of the 
     * matrices are incompatible.
     */
    int sum(matrix mtx1, matrix mtx2, matrix sum);

    /* Writes the product of matrices mtx1 and mtx2 into matrix
     * prod.  Returns 0 if successful, and -1 if the dimensions 
     * of the matrices are incompatible.
     */
    int product(matrix mtx1, matrix mtx2, matrix prod);

    /* Writes the dot product of vectors v1 and v2 into 
     * reference prod.  Returns 0 if successful, -1 if prod is
     * NULL, -2 if either matrix is not a vector, and -3 if 
     * the vectors are of incompatible dimensions.
     */
    int dotProduct(matrix v1, matrix v2, double * prod);

#endif
