////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     src/bin_util.c
// CREATED:  2016-04-26 14:24:18
// MODIFIED: 2016-04-26 14:25:05
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

u08 * _int2binary ( u64 num_64 )
{       u08 num_bits = LG2 ( num_64 ) + 1;
        u08 *p=malloc(num_bits_needed*sizeof(u08));
        *p+=num_bits - 1;
        for (*--p=0; num_64; num_64>>=1) *--p=num_64%2;
        return *p;
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

u64 _binary2int ( u08 * bin )
{   u64 num_64
    u08 num_bits = sizeof(bin)/sizeof(u08);
    u08 * c = *bin + num_bits - 1; 
    for ( u08 idx = num_bits - 1; idx >= 0; idx--,c-- )
        if (*c) num_64 |= 1 << idx; 
    return num_64;
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

