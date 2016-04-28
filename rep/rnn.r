////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.r
// CREATED:  2016-04-23 21:22:35
// MODIFIED: 2016-04-28 03:08:51
////////////////////////////////////////////////////////////////////
#ifndef RNN_R
    #define RNN_R
    #include "../inc/types.h"
    #include "../inc/matrix.h"
    #include <stdargs.h>
    #include <stdlib.h>
    #include <gsl/gsl_matrix.h>

    typedef struct Synapse  Synapse;
    typedef struct Layers   Layers
    typedef struct RNN      RNN;

    struct Synapse {
        gsl_matrix *    _0;
        gsl_matrix *    _0_update;
        gsl_matrix *    _1;
        gsl_matrix *    _1_update;
        gsl_matrix *    _h;
        gsl_matrix *    _h_update;
    };
    
    struct Layers {
        gsl_matrix *    _1;
        gsl_matrix *    _1_values
        gsl_matrix *    _1_prev;
        gsl_matrix *    _1_future_delta;
        gsl_matrix *    _2;
        gsl_matrix *    _2_deltas;
    };

    struct RNN {
        const void * class;         //must be first
        void * ( update )           ( void * _self, u64 inn, u64 * inl );    
        void * ( train )            ( void * _self, u32 outn, u64 * outl, u64 inn, u64 * inl );
        void * ( toggle_train )     ( void * _self, u08 flag );
        void * ( _init_synap )      ( void * _self );
        void * ( _init_layer )      ( void * _self );
        void * ( _kill_synap )      ( void * _self );
        void * ( _kill_layer )      ( void * _self );
        
        va_list*    _options;

        // input variables - - - - - - - - -
        double      alpha_f;
        u32         in_dim_64;
        u32         hidden_dim_64;
        u32         out_dim_64;
        u32         bin_dim_64;
        u32         num_out_32;
        u32         num_in_32;
        // - - - - - - - - - - - - - - - - - 

        Synapse         synap;
        Layers          layer;
        
        u08             _train_flg;

        u64  *          inputs_64;
        u64  *          outputs_64;
        gsl_matrix *    inputs_bl;
        gsl_matrix *    outputs_bl;
        gsl_matrix *    prediction_bl;
        gsl_matrix *    difference_bl;
        gsl_matrix *    real_answer_bl;
        double          overall_error_f;
        
    }

#endif
