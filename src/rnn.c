////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-26 19:28:51
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
#include "../inc/matrix.h"

#include "../rep/class.r"
#include "../rep/rnn.r"

/// __init()__ called with:
/// new ( RNN , double alpha, u64 in_dim, u64 hidden_dim, u64 out_dim, u32 num_in )
void * RNN_init ( const void * _self , va_list *args )
{   struct RNN  * self = _self;
    
    #ifdef _VERBOSE
        TRACE("Creating recurrent neural network ... ", "RNN_init");
    #endif
    
    va_copy ( self -> _options, args );

    // THE ALPHA OFFSET 
    self->alpha_f = va_arg ( * args , double );
    assert ( self -> alpha_f );
    #ifdef _VERBOSE
        TRACE("Alpha = %f \n" self -> alpha_f);
    #endif
    
    // GET THE INPUT DIMENSION 
    self->in_dim_64 = va_arg ( * args , u64 );
    assert ( self -> in_dim_64 );
    #ifdef _VERBOSE
        TRACE("Input Dimension = %llu \n" self -> in_dim_64);
    #endif

    // GET LENGTH OF HIDDEN ARRAY
    self->hidden_dim_64 = va_arg ( * args , u64 );
    assert ( self -> hidden_dim_64 );
    #ifdef _VERBOSE
        TRACE("Hidden Dimension = %llu \n" self -> hidden_dim_64);
    #endif
    
    // GET LENGTH OF OUTPUT ARRAY
    self->out_dim_64 = va_arg ( * args , u64 );
    assert ( self -> out_dim_64 );
    #ifdef _VERBOSE
        TRACE("Output Dimension = %llu \n" self -> out_dim_64);
    #endif

    // GET LENGTH OF BINARY INPUT ARRAY
    self->num_in_32 = va_arg ( * args , u64 );
    assert ( self -> bin_dim_64 );
    #ifdef _VERBOSE
        TRACE("Binary Dimension = %u \n" self -> num_in_32);
    #endif

    // GET NUMBER OF OUTPUTS
    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_out_32 );
    #ifdef _VERBOSE
        TRACE("Number of outputs = %u \n" self -> num_out_32);
    #endif

    // GET NUMBER OF INPUTS
    self->num_in_32 = va_arg ( * args , u32 );
    assert ( self -> num_in_32 );
    #ifdef _VERBOSE
        TRACE("Number of Inputs = %u \n" self -> num_in_32);
        TRACE("Initializing RNN internal methods\n","RNN");
    #endif

    // INTERNAL METHODS
    self->input = _input;
    self->train = _train;
    self->toggle_train = _toggle_train;
    self->_init_synap = __init_synap;
    self->_kill_synap = __kill_synapse;
    self->_init_layer = __init_layer;
    self->_kill_layer = __kill_layers;
    
    
    // INTERNAL STRUCT INIT FUNCTIONS
    self->synap = self->_init_synap(self);
    self->layer = self->_init_layers(self);
    
    // Initialize internal Vars and Counters
    self.train_flag         = TRUE;
    self.overall_error_f    = 0.0;
    return self;
}// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void * RNN_del ( void * self ) 
{   struct RNN * self = _self;
    #ifdef _VERBOSE
        TRACE("Deleting RNN %s","\n");
    #endif
    self->_kill_synapse(self);
    self->_kill_layers(self);
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

void _toggle_train (void * _self ) 
{   struct RNN * self = _self;
    #ifdef _VERBOSE
    TRACE("training mode toggled to: %s\n",
            (self->train_flag)?"OFF":"ON");
    #endif
    self->train_flag = !(self.train_flag);
    return;
}
Layers  * __init_layer ( void * _self )
{   struct RNN * self = _self;
    #ifdef _VERBOSE
        TRACE("layer._1 initializing..","\n");
    #endif
    
}
Synapse * __init_synap ( void * _self )
{   struct RNN * self = _self; 

    //synap._0 -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._0 = [ %d , %d ]", self->in_dim_64, self->hidden_dim_64);
    #endif
        self->synap->_0 = newMatrix(self->in_dim_64,self->hidden_dim_64); 
    for ( u64 i = 0; i < self->in_dim_64; i++ )
        for( u64 j = 0; j < self->hidden_dim_64; j++)
            setElement( self->synap->_0, i, j, (double)rand()/(double)RAND_MAX );    
    
}
    //synap._1 -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._1 = [ %d , %d ]", self->hidden_dim_64, self->out_dim_64);
    #endif

    for ( u64 i = 0; i < self->hidden_dim_64; i++ )
        for( u64 j = 0; j < self->hidden_dim_64; j++)
            setElement( self->synap->_0, i, j, (double)rand()/(double)RAND_MAX );    

    //synap._h -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._h = [ %d , %d ]", self->hidden_dim_64, self->hidden_dim_64);
    #endif

    for ( u64 i = 0; i < self->hidden_dim_64; i++ )
        for( u64 j = 0; j < self->hidden_dim_64; j++)
            setElement( self->synap->_0, i, j, (double)rand()/(double)RAND_MAX );    

    return self->synap;
}

void __kill_synap ( void * self )
{   struct RNN * self = _self;
    truefree ( self->synap->_0 );
    truefree ( self->synap->_h );
    truefree ( self->synap->_1 );
    truefree ( self->synap );
}
