////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     src/matrix_util.c
// CREATED:  2016-04-26 14:22:16
// MODIFIED: 2016-04-26 14:28:15
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

double _sigmoid ( double num_f )
{   return 1f/(1+exp(-num_f));
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

double _sigmoid_derivative ( double num_f )
{   return num_f*(1-num_f);
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void _transpose(double m[], const unsigned h, const unsigned w)
{   for (unsigned start = 0; start <= w * h - 1; ++start)
    {
        unsigned next = start;
        unsigned i = 0;
        do
        {   ++i;
            next = (next % h) * w + next / h;
        } while (next > start);

        if (next >= start && i != 1)
        {   const double tmp = m[start];
            next = start;
            do
            {   i = (next % h) * w + next / h;
                m[next] = (i == start) ? tmp : m[i];
                next = i;
            } while (next > start);
        }
    }   
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
