////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     inc/matrix.h
// CREATED:  2016-04-26 17:31:45
// MODIFIED: 2016-04-26 21:17:13
////////////////////////////////////////////////////////////////////
#ifndef INC/MATRIX_H
    #define INC/MATRIX_H
    #include <stddef.h>

    struct _matrix * matrix;
    matrix newMatrix(u32 rows, u32 cols);
    matrix copyMatrix(matrix mtx);
    void deleteMatrix(matrix mtx);

    u32 setElement(matrix mtx, u32 row, u32 col, double val);
    u32 setRow(matrix mtx, u32 row, double * vector);
    
    u32 getRow(matrix mtx, u32 row, double * vector);
    u32 getCol(matrix mtx, u32 col, double * vector);
    u32 getElement(matrix mtx, u32 row, u32 col, double * val);
    
    u32 nRows(matrix mtx);
    u32 nCols(matrix mtx);
    
    u32 transpose(matrix in, matrix out);
    u32 sum(matrix mtx1, matrix mtx2, matrix sum);
    u32 product(matrix mtx1, matrix mtx2, matrix prod);
    u32 dotProduct(matrix v1, matrix v2, double * prod);
    
    void printMatrix(matrix mtx);

#endif
