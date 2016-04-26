////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-26 14:25:08
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

    // THE ALPHA OFFSET 
    self->alpha_f = va_arg ( * args , double );
    assert ( self -> alpha_f );
    #ifdef DEBUG
        TRACE("Alpha = %f \n" self -> alpha_f);
    #endif
    
    // GET THE INPUT DIMENSION 
    self->in_dim_64 = va_arg ( * args , u64 );
    assert ( self -> in_dim_64 );
    #ifdef DEBUG
        TRACE("Input Dimension = %llu \n" self -> in_dim_64);
    #endif

    // GET LENGTH OF HIDDEN ARRAY
    self->hidden_dim_64 = va_arg ( * args , u64 );
    assert ( self -> hidden_dim_64 );
    #ifdef DEBUG
        TRACE("Hidden Dimension = %llu \n" self -> hidden_dim_64);
    #endif
    
    // GET LENGTH OF OUTPUT ARRAY
    self->out_dim_64 = va_arg ( * args , u64 );
    assert ( self -> out_dim_64 );
    #ifdef DEBUG
        TRACE("Output Dimension = %llu \n" self -> out_dim_64);
    #endif

    // GET LENGTH OF BINARY INPUT ARRAY
    self->num_in_32 = va_arg ( * args , u64 );
    assert ( self -> bin_dim_64 );
    #ifdef DEBUG
        TRACE("Binary Dimension = %u \n" self -> num_in_32);
    #endif

    // GET NUMBER OF OUTPUTS
    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_out_32 );
    #ifdef DEBUG
        TRACE("Number of outputs = %u \n" self -> num_out_32);
    #endif

    // GET NUMBER OF INPUTS
    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_in_32 );
    #ifdef DEBUG
        TRACE("Number of Inputs = %u \n" self -> num_in_32);
    #endif

    // INTERNAL METHODS
    self->input = _input;
    self->train = _train;
    self->_init_synap = __init_synap;
    self->_kill_synap = __kill_synap;
    self->_init_layer = __init_layer;
    self->_kill_layer = __kill_layer;
    
    // INTERNAL STRUCT INIT FUNCTIONS
    self->synap = self->_init_synap(self);
    self->layer = self->_init_layers(self);
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

Synapse * __init_synap ( void * _self ){
    struct RNN * self = _self; 
    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._0 = [ %d , %d ]", self->in_dim_64, self->hidden_dim_64);
    #endif
    assert(self->synap._0 = malloc(self->in_dim_64*sizeof(double *)));
    assert(self->synap._0_update = malloc(self->in_dim_64*sizeof(double *)));

    for ( u64 i = 0; i < self->in_dim_64; i++ )
    {   assert ( self->synap._0[i] = malloc(self->hidden_dim_64*sizeof(double)));
        assert ( self->synap._0_update[i] = calloc(self->hidden_dim_64,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }

    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._1 = [ %d , %d ]", self->hidden_dim_64, self->out_dim_64);
    #endif
    assert(self->synap._1 = malloc(self->hidden_dim_64*sizeof(double *)));
    assert(self->synap._1_update = malloc(self->out_dim_64*sizeof(double *)));

    for ( u64 i = 0; i < self->hidden_dim_64; i++ )
    {   assert ( self->synap._0[i] = malloc(self->out_dim_64*sizeof(double)));
        assert ( self->synap._0_update[i] = calloc(self->out_dim_64,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }

    //--------------------------------------------------------
    #ifdef DEBUG
        TRACE("synap._h = [ %d , %d ]", self->hidden_dim_64, self->hidden_dim_64);
    #endif
    assert(self->synap._0 = malloc(self->hidden_dim_64*sizeof(double *)));
    assert(self->synap._0_update = malloc(self->hidden_dim_64*sizeof(double *)));

    for ( u64 i = 0; i < self->hidden_dim_64; i++ )
    {   assert(self->synap._0[i] = malloc(self->hidden_dim_64*sizeof(double)));
        assert(self->synap._0_update[i] = calloc(self->hidden_dim_64,sizeof(double)));
        FOREACH(double *syn IN self->synap._0[i])    
            *syn = ( 2f * rand() ) - 1;    
    }

    return self->synap;
}

void _kill_synap ( void * self )
{   struct RNN * self = _self;
    free ( self->synap->_0 );
    free ( self->synap->_h );
    free ( self->synap->_1 );
}
