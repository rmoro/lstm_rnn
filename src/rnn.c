////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com 
// FILE:     rnn.c
// CREATED:  2016-04-23 21:56:51
// MODIFIED: 2016-04-24 01:05:14
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

double sigmoid ( double x )
{   return 1f/(1+exp(-x));
}

double sigmoid_derivative ( double x );
{   return x*(1-x);
}

void int2binary ( bool * bin , u64 num64 )
{   // TODO: IMPLEMENT THIS FUNCTION
    return;
}

u64 binary2int ( bool * bin )
{   // TODO: IMPLEMENT THIS FUNCTION
    return;
}

void transpose ( bool * src, bool * dest, u64 dest_len);
{   long i, j, *row, temp;

   for(i=0; i<(1<<BL); i++) {
      row=startpt;
      for(j=0; j<(1<<BL); j+=8, row+=8*(1<<BL)) {

         temp = destpt[j];
         destpt[j] = row[0];
         row[0] = temp;
                                          /* gcc turns the BL stuff here  */
         temp = destpt[j+1];              /* into numbers, saving lots of */
         destpt[j+1] = row[1<<BL];        /* pointer addition and AGIs    */
         row[1<<BL] = temp;

         temp = destpt[j+2];
         destpt[j+2] = row[2*(1<<BL)];           
         row[2*(1<<BL)] = temp;

         temp = destpt[j+3];
         destpt[j+3] = row[3*(1<<BL)];           
         row[3*(1<<BL)] = temp;

         temp = destpt[j+4];
         destpt[j+4] = row[4*(1<<BL)];
         row[4*(1<<BL)] = temp;
                                    
         temp = destpt[j+5];        
         destpt[j+5] = row[5*(1<<BL)];
         row[5*(1<<BL)] = temp;

         temp = destpt[j+6];
         destpt[j+6] = row[6*(1<<BL)];           
         row[6*(1<<BL)] = temp;

         temp = destpt[j+7];
         destpt[j+7] = row[7*(1<<BL)];           
         row[7*(1<<BL)] = temp;
      }
      startpt++;                                 /* move to next column */
      destpt+=destlength;                        /* move to next row    */
   }                     
}
