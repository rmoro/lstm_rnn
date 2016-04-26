////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     inc/matrix_util.h
// CREATED:  2016-04-26 14:25:58
// MODIFIED: 2016-04-26 14:28:14
////////////////////////////////////////////////////////////////////
#ifndef INC/MATRIX_UTIL_H
    #define INC/MATRIX_UTIL_H
    #include <stddef.h>

     
    double  sigmoid ( double num_f );
    double  sigmoid_derivative ( double num_f );
    void    transpose(double m[], const unsigned h, const unsigned w ); 

#endif
