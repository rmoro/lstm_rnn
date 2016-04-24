////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.h
// CREATED:  2016-04-23 20:50:46
// MODIFIED: 2016-04-24 00:35:37
////////////////////////////////////////////////////////////////////
#ifndef RNN_H
    #define RNN_H
    #include <stddef.h>

    double sigmoid ( double x );
    double sigmoid_derivative ( double x );
    void print_state ( RNN * _self );
    void int2binary ( bool * bin , u64 num64 );
    u64 binary2int ( bool * bin );
    void transpose ( bool * src, bool * dest, u64 dest_len);

#endif
