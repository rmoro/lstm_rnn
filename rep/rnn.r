////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.r
// CREATED:  2016-04-23 21:22:35
// MODIFIED: 2016-04-26 14:20:40
////////////////////////////////////////////////////////////////////
#ifndef RNN_R
    #define RNN_R
    #include "../inc/types.h"
    #include <stdargs.h>
    #include <stdlib.h>

    typedef struct Synapse  Synapse;
    typedef struct Layers   Layers
    typedef struct RNN      RNN;

    struct Synapse {
        double **   _0;
        double      _0_update;
        double **   _1;
        double      _1_update;
        double **   _h;
        double      _h_update;
    };
    
    struct Layers {
        double *    _0;
        double *    _1;
        double **   _1_values
        double *    _1_prev;
        double *    _1_future_delta;
        double *    _2;
        double **   _2_deltas;
    };

    struct RNN {
        const void * class; //must be first
        void * ( update )           ( void * _self, u64 inn, u64 * inl );    
        void * ( train )            ( void * _self, u32 outn, u64 * outl, u64 inn, u64 * inl );
        void * ( _init_synap )      ( void * _self );
        void * ( _init_layer )      ( void * _self );
        void * ( _kill_synap )      ( void * _self );
        void * ( _kill_layer )      ( void * _self );
        va_list*     _options;

        // input variables - - - - - - - - -
        double      alpha_f;
        u64         in_dim_64;
        u64         hidden_dim_64;
        u64         out_dim_64;
        u64         bin_dim_64;
        u32         num_out_32;
        u32         num_in_32;
        // - - - - - - - - - - - - - - - - - 

        Synapse     synap;
        Layers      layer;

        u64  *      inputs_64;
        u08  **     inputs_bl;
        u64  *      outputs_64;
        u08  **     outputs_bl;
        u08  *      prediction_bl;
        u08  *      difference_bl;
        
        double      overall_error_f;
        
    }

#endif
