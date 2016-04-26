////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-25 22:07:34
////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <math.h>
#include <random.h>
#include <stdargs.h>
#include <stdio.h>
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
{   struct RNN  * self = _self;
    
    #ifdef DEBUG
        TRACE("Creating recurrent neural network ... ", "RNN_init");
    #endif
    
    va_copy ( self -> _options, args );

    self->alpha_f = va_arg ( * args , double );
    assert ( self -> alpha_f );
    #ifdef DEBUG
        TRACE("Alpha = %f \n" self -> alpha_f);
    #endif

    self->in_dim_64 = va_arg ( * args , u64 );
    assert ( self -> in_dim_64 );
    u64 id_t = self->in_dim_64;
    #ifdef DEBUG
        TRACE("Input Dimension = %llu \n" self -> in_dim_64);
    #endif

    self->hidden_dim_64 = va_arg ( * args , u64 );
    assert ( self -> hidden_dim_64 );
    u64 hd_t = self->hidden_dim_64;
    #ifdef DEBUG
        TRACE("Hidden Dimension = %llu \n" self -> hidden_dim_64);
    #endif

    self->out_dim_64 = va_arg ( * args , u64 );
    assert ( self -> out_dim_64 );
    u64 od_t = self->out_dim_64;
    #ifdef DEBUG
        TRACE("Output Dimension = %llu \n" self -> out_dim_64);
    #endif

    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_in_32 );
    #ifdef DEBUG
        TRACE("Number of Inputs = %u \n" self -> num_in_32);
    #endif
    
    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._0 = [ %d , %d ]", id_t, hd_t);
    #endif
    assert(self->synap._0 = malloc(id_t*sizeof(double *)));
    assert(self->synap._0_update = malloc(id_t*sizeof(double *)));

    for ( u64 i = 0; i < id_t; i++ )
    {   assert ( self->synap._0[i] = malloc(hd_t*sizeof(double)));
        assert ( self->synap._0_update[i] = calloc(hd_t,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }

    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._1 = [ %d , %d ]", hd_t, od_t);
    #endif
    assert(self->synap._1 = malloc(hd_t*sizeof(double *)));
    assert(self->synap._1_update = malloc(od_t*sizeof(double *)));

    for ( u64 i = 0; i < hd_t; i++ )
    {   assert ( self->synap._0[i] = malloc(od_t*sizeof(double)));
        assert ( self->synap._0_update[i] = calloc(od_t,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }

    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._h = [ %d , %d ]", hd_t, hd_t);
    #endif
    assert(self->synap._0 = malloc(hd_t*sizeof(double *)));
    assert(self->synap._0_update = malloc(hd_t*sizeof(double *)));

    for ( u64 i = 0; i < hd_t; i++ )
    {   assert(self->synap._0[i] = malloc(hd_t*sizeof(double)));
        assert(self->synap._0_update[i] = calloc(hd_t,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }
    
    return self;
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void * RNN_del ( void * self ) 
{   struct RNN * self = _self;
    free(self->synap._0);
    free(self->synap._1);
    free(self->synap._h);
    return free (self);
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void * RNN_cpy (void * _self ) 
{   const struct RNN * self = _self;
    return new ( RNN , self->_options );
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void * RNN_str (void * self ) 
{   const struct RNN * self = _self;
    
    //TODO: Implement __str__(RNN)
    
    return; //pstring;
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

double _sigmoid ( double num_f )
{   return 1f/(1+exp(-num_f));
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

double _sigmoid_derivative ( double num_f );
{   return num_f*(1-num_f);
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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
