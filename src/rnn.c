////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-30 20:32:45
////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <math.h>
#include <random.h>
#include <stdargs.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

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
    self->in_dim_64 = va_arg ( * args , u32 );
    assert ( self -> in_dim_64 );
    #ifdef _VERBOSE
        TRACE("Input Dimension = %llu \n" self -> in_dim_64);
    #endif

    // GET LENGTH OF HIDDEN ARRAY
    self->hidden_dim_64 = va_arg ( * args , u32 );
    assert ( self -> hidden_dim_64 );
    #ifdef _VERBOSE
        TRACE("Hidden Dimension = %llu \n" self -> hidden_dim_64);
    #endif
    
    // GET LENGTH OF OUTPUT ARRAY
    self->out_dim_64 = va_arg ( * args , u32 );
    assert ( self -> out_dim_64 );
    #ifdef _VERBOSE
        TRACE("Output Dimension = %llu \n" self -> out_dim_64);
    #endif

    // GET LENGTH OF BINARY INPUT ARRAY
    self->num_in_32 = va_arg ( * args , u32 );
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


/// Training function.  Function is feed inputs and outputs in binary vectors. 
//  each training cycle updates the RNN.  Will only work if train_toggle == true;
void 
__train ( 
        void        * _self,    // link to class 
        u32         inn,        // number of inputs
        gsl_vector  inv[],      // list of input vectors
        u32         outn,       // number of outputs
        gsl_vector  outv[])     // output vector list
{   
    struct RNN * self = _self;
    self->_reset_layers(self);
    self->overall_error_f = 0.0;

    gsl_vector prediction = gsl_vector_alloc(outv[1]->size);

    u32 prev_layer_index = 0;
    gsl_vector * layer_1 = gsl_vector_calloc ( self->synap->_0->size2 );
    for( u32 position = 0; position < self->bin_dim_32; position++)
    {   // create the input and output vectors
        //
        gsl_vector *input_v = gsl_vector_alloc (inn);
        for ( u32 i = 0 ; i < inn; i++)
            gsl_vector_set ( input_v, i, inv[i][(self->bin_dim_32-1) - position] );
            // ^^^ replace with gsl_vector_get(inv[i], (self->bin_dim -1) - position)
            //
        #ifdef _VERBOSE
            TRACE("created input vector %s","\n");
        #endif
        gsl_vector *output_v = gsl_vector_alloc (outn);
        for ( u32 i = 0; i < outn; i++ )
            gsl_vector_set ( output_v, i, outv[i][(self->bin_dim-1)-position]);
            // ^^^ replace with gsl_vector_get(inv[i], (self->bin_dim -1) - position)
            //
        #ifdef _VERBOSE
            TRACE("created output vector %s","\n");
        #endif
        for ( u32 i = 0; i < inn; i++ )
        {   double s1 = gsl_vector_get ( input_v , i );
            gsl_vector * t1 = gsl_matrix_row ( self->synap->_0 , i );
            gsl_vector_scale ( t1 , s1 );
            gsl_vecor_add ( layer_1 , t1 );
            gsl_vector_free ( t1 );
        }
        #ifdef _VERBOSE
            TRACE("numpy.dot ( input vector , synapse 0 )%s","\n");
        #endif

        sigmoid ( layer_1 );
        #ifdef _VERBOSE
            TRACE("sigmoid ( layer_1 ); %s","\n");
        #endif

        u32 h = self->synap->_h->size1;
        for ( u32 r = 0; r < h; r++)
        {   gsl_vector * tv2 = gsl_matrix_row ( self-synap->_h, r );
            gsl_vector * tv  = gsl_matrix_row ( self->layer->_1_values, \
                    self->layer->_1_values->size1 - position - 1);
            gsl_vector_mul ( tv , tv2 );
            gsl_vector_add ( layer_1, tv );
            gsl_free ( tv ); gsl_free (tv2);
        }
            
        #ifdef _VERBOSE
            TRACE("numpy.dot ( previous_layer, hidden_synapse ) %s","\n");
        #endif
        
        gsl_matrix * layer_2 = calloc ( h, h ); 
        for ( u32 i = 0; i < h; i++)
        {   for ( u32 j = 0; j < h; j++)
            {   double product = \
                 gsl_matrix_get ( self->synap->_1, i, 0 ) * \
                 gsl_vector_get ( layer_1 , j );
                 gsl_matrix_set ( layer_2 , i, j, product );
            }
        }

        #ifdef _VERBOSE
            TRACE("layer 2 = dot ( synap 1 , layer 1 ) %s","\n");
        #endif
        
        sigmoid ( layer_2 );

        #ifdef _VERBOSE
            TRACE("sigmoid ( layer_2 ); %s","\n");
        #endif


void 
_toggle_train (void * _self ) 
{   struct RNN * self = _self;
    #ifdef _VERBOSE
    TRACE("training mode toggled to: %s\n",
            (self->train_flag)?"OFF":"ON");
    #endif
    self->train_flag = !(self.train_flag);
    return;
}

Layers  * 
__init_layer ( void * _self )
{   struct RNN * self = _self;
    #ifdef _VERBOSE
        TRACE("layer._1 initializing..","\n");
    #endif
    self->layer->_1 
}

Synapse * 
__init_synap ( void * _self )
{   struct RNN * self = _self; 

    //synap._0 -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._0 = [ %d , %d ]", self->in_dim_64, self->hidden_dim_64);
    #endif
    self->synap->_0 = gsl_matrix_alloc (self->in_dim_64,self->hidden_dim_64); 
    for ( u32 i = 0; i < self->in_dim_64; i++ )
        for( u32 j = 0; j < self->hidden_dim_64; j++)
            gsl_matrix_set ( self->synap->_0, i, j, (double)rand()/(double)RAND_MAX );    
    
    //synap._1 -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._1 = [ %d , %d ]", self->hidden_dim_64, self->out_dim_64);
    #endif
    self->synap->_1 = gsl_matrix_alloc (self->hidden_dim_64,self->out_dim_64); 
    for ( u32 i = 0; i < self->hidden_dim_64; i++ )
        for( u32 j = 0; j < self->out_dim_64; j++)
            gsl_matrix_set ( self->synap->_1, i, j, (double)rand()/(double)RAND_MAX );    

    //synap._h -----------------------------------------------
    #ifdef _VERBOSE
        TRACE("synap._h = [ %d , %d ]", self->hidden_dim_64, self->hidden_dim_64);
    #endif
    self->synap->_h = gsl_matrix_alloc (self->hidden_dim_64,self->hidden_dim_64); 
    for ( u32 i = 0; i < self->hidden_dim_64; i++ )
        for( u32 j = 0; j < self->hidden_dim_64; j++)
            gsl_matrix_set ( self->synap->_h, i, j, (double)rand()/(double)RAND_MAX );    

    return self->synap;
}

void 
__kill_synap ( void * self )
{   struct RNN * self = _self;
    truefree( self->synap->_0 );
    truefree( self->synap->_h );
    truefree( self->synap->_1 );
    truefree( self->synap );
}
