////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com
// FILE:     types.h
// CREATED:  2016-04-21 12:03:42
// MODIFIED: 2016-04-28 11:10:06
////////////////////////////////////////////////////////////////////

#ifndef INTDEF
    #include <stdint.h>
    #define INTDEF
    #ifdef uint128_t
        typedef uint128_t       u128;
    #endif
    typedef uint64_t            u64;
    typedef uint32_t            u32;
    typedef uint16_t            u16;
    typedef uint8_t             u08;
    
    #ifdef int128_t
        typedef int_128_t       s128;
    #endif
    typedef int64_t             s64;
    typedef int32_t             s32;
    typedef int16_t             s16;
    typedef int8_t              s08;
#endif

#ifndef GSL
    #define GSL
    #include <gsl/gsl_matrix.h>

    typedef gsl_matrix matrix_d     
    // putting this here in case i decide 
    // to get more specific about the type 
    // of matrix

#endif

#ifndef TUPLE_32
#define TUPLE_32
    typedef struct {
        u32             first;
        u32             second;
    } u32Tuple;
#endif


