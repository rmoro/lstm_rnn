////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     src/matrix.c
// CREATED:  2016-04-26 17:34:57
// MODIFIED: 2016-04-26 22:14:18
////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

struct _matrix {
  u32 rows;
  u32 cols;
  double ** data;
};

matrix newMatrix(u32 rows, u32 cols) {
  // allocate a matrix structure
  matrix m = (matrix) malloc(sizeof(struct _matrix));

  // set dimensions
  m->rows = rows;
  m->cols = cols;

  if (rows > 0 && cols > 0) {
    // allocate a double array of length rows * cols
    m->data = malloc(rows*sizeof( double * ));
    for( i = 0; i < rows; i++ )
        m->data[i] = malloc(cols*sizeof(double));

    // set all data to 0
    u32 i;
    for (i = 0; i < rows*cols; i++)
        for ( u32 j = 0; j < cols; j++ ) 
            m->data[i] = 0.0;
  }
  else
    m->data = NULL;
 
  return m;
}

matrix copyMatrix(matrix mtx) {
  matrix cp = NULL;

  if (!mtx) return NULL;

  // create a new matrix to hold the copy
  cp = newMatrix(mtx->rows, mtx->cols);
  assert (cp);

  // copy mtx's data to cp's data
  if (mtx->data)
    memcpy(cp->data, mtx->data, 
           mtx->rows * mtx->cols * sizeof(double));

  return cp;
}

void deleteMatrix(matrix mtx) {
  if (mtx->data) free(mtx->data);
  free(mtx);
}

#define ELEM(mtx, row, col) mtx->data[row][col]

u32 setRow(matrix mtx, u32 row, double * row) {
    
    if ( row >= rows ) 
    {   TRACE("[ FATAL ERROR ] row overflow in matrix, row = %u\n",row);
        return -3;
    }
        
    
    if ( sizeof( mtx->data[ row ] ) != sizeof( row ) )
    {   TRACE("[ FATAL ERROR ] wrong sized row given to setRow(..)%s","\n");
        return -2;
    }
    // TODO: Decide if this function 
    // should return the previous row
    mtx->data[row] = NULL;
    
    #ifdef _VERBOSE
        TRACE("GETTING ROW #%u\n",row);
    #endif
    mtx->data[row] = row;
    
        
    if ( mtx->data[row] == NULL ) 
    {   TRACE("[ FATAL ERROR ] error allocating memory for row%s","\n")
        return -1;
    }

    truefree ( row );

   return 0;
}

u32 setElement(matrix mtx, u32 row, u32 col, double val) {
  if (row <= 0 || row > mtx->rows ||
      col <= 0 || col > mtx->cols)
    return -1;

  ELEM(mtx, row, col) = val;
  return 0;
}

u32 getElement(matrix mtx, u32 row, u32 col, double * val) {
  if (!val) return -1;
  if (row <= 0 || row > mtx->rows ||
      col <= 0 || col > mtx->cols)
    return -2;

  *val = ELEM(mtx, row, col);
  return 0;
}
u32 getRow(matrix mtx, u32 row, double * row_ptr){
    if(row >= mtx->rows) 
    {   TRACE("[ FATAL ERROR ] ROW NOT WITHIN MATRIX%s","\n");
        return -2;
    }
    row_ptr = NULL;
    row_prt = malloc(sizeof(mtx->data[row]));
    if ( row == NULL ) 
    {   TRACE("[ FATAL ERROR ] error allocating memory for row%s","\n");
        return -1;
    }
    row_ptr = mtx->data[row];
    return 0;
}   

u32 nRows(matrix mtx) {
  return mtx->rows;
}

u32 nCols(matrix mtx) {
  return mtx->cols;
}

void printMatrix(matrix mtx) {
  u32 row, col;
  for (row = 1; row <= mtx->rows; row++) {
    for (col = 1; col <= mtx->cols; col++)
      printf("% 6.2f ", ELEM(mtx, row, col));
    printf("\n");
  }
}

u32 transpose(matrix in, matrix out) {
  if (in->rows != out->cols || in->cols != out->rows)
    return -2;

  u32 row, col;
  for (row = 1; row <= in->rows; row++)
    for (col = 1; col <= in->cols; col++)
      ELEM(out, col, row) = ELEM(in, row, col);
  return 0;
}

u32 sum(matrix mtx1, matrix mtx2, matrix sum) {
  if (!mtx1 || !mtx2 || !sum) return -1;
  if (mtx1->rows != mtx2->rows ||
      mtx1->rows != sum->rows ||
      mtx1->cols != mtx2->cols ||
      mtx1->cols != sum->cols)
    return -2;

  u32 row, col;
  for (col = 1; col <= mtx1->cols; col++)
    for (row = 1; row <= mtx1->rows; row++)
      ELEM(sum, row, col) = 
        ELEM(mtx1, row, col) + ELEM(mtx2, row, col);
  return 0;
}

u32 product(matrix mtx1, matrix mtx2, matrix prod) {

  if (!mtx1 || !mtx2 || !prod) return -1;
  if (mtx1->cols != mtx2->rows ||
      mtx1->rows != prod->rows ||
      mtx2->cols != prod->cols)
    return -2;

  u32 row, col, k;
  for (col = 1; col <= mtx2->cols; col++)
    for (row = 1; row <= mtx1->rows; row++) {
      double val = 0.0;
      for (k = 1; k <= mtx1->cols; k++)
        val += ELEM(mtx1, row, k) * ELEM(mtx2, k, col);
      ELEM(prod, row, col) = val;
    }
  return 0;
}

u32 dotProduct(matrix v1, matrix v2, double * prod) {
  if (!prod) return -1;
  if (v1->cols != 1 || v2->cols != 1) return -2;
  if (v1->rows != v2->rows) return -3;

  *prod = 0;
  u32 i;
  for (i = 1; i <= v1->rows; i++)
    *prod += ELEM(v1, i, 1) * ELEM(v2, i, 1);
  return 0;
}
