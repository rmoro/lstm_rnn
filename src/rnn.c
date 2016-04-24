////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-23 22:45:48
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <stdargs.h>
#include <stdlib.h>

#include "../inc/macros.h"
#include "../inc/types.h"
#include "../inc/class.h"
#include "../inc/rnn.h"

#include "../rep/class.r"
#include "../rep/rnn.r"

/// __init()__ called with:
/// new ( RNN , double alpha, u64 in_dim, u64 hidden_dim, u64 out_dim, u32 num_in )
void * RNN_init ( const void * _self , va_list *args )
{
    struct RNN  * self = _self;
    
    #ifdef DEBUG
        TRACE("Creating recurrent neural network ... ", "RNN_init");
    #endif

    self->alpha_f = va_arg ( * args , double );
    assert ( self -> alpha_f );
    #ifdef DEBUG
        TRACE("Alpha = %f \n" self -> alpha_f);
    #endif

    self->in_dim_64 = va_arg ( * args , u64 );
    assert ( self -> in_dim_64 );
    #ifdef DEBUG
        TRACE("Input Dimension = %llu \n" self -> in_dim_64);
    #endif

    self->hidden_dim_64 = va_arg ( * args , u64 );
    assert ( self -> hidden_dim_64 );
    #ifdef DEBUG
        TRACE("Hidden Dimension = %llu \n" self -> hidden_dim_64);
    #endif

    self->out_dim_64 = va_arg ( * args , u64 );
    assert ( self -> out_dim_64 );
    #ifdef DEBUG
        TRACE("Output Dimension = %llu \n" self -> out_dim_64);
    #endif

    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_in_32 );
    #ifdef DEBUG
        TRACE("Number of Inputs = %u \n" self -> num_in_32);
    #endif

    

    

