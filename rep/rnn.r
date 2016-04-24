////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.r
// CREATED:  2016-04-23 21:22:35
// MODIFIED: 2016-04-23 22:45:40
////////////////////////////////////////////////////////////////////
#ifndef RNN_R
    #define RNN_R
    #include "../inc/types.h"

    typedef struct Synapse Synapse;
    
    struct Synapse {
        double  _0;
        double  _1;
        double  _h;
    };

    struct RNN {
        const void * class; //must be first
        
        // input variables - - - - - - - - -
        double      alpha_f;
        u64         in_dim_64;
        u64         hidden_dim_64;
        u64         out_dim_64;
        u32         num_in_32;
        // - - - - - - - - - - - - - - - - - 

        Synapse     synap;
        
        u64 *       inputs_64;
        bool **     inputs_bl;

        u64         output_64;
        bool *      output_bl;

        bool *      prediction_bl;

        double      overall_error_f;
        
        //TODO: not sure of the types for these
        double      _layer0; //input layer
        double      _layer1; //hidden layer
        double      _layer2; //output layer
    
    }

#endif
