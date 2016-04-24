////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com
// FILE:     class.c
// CREATED:  2016-04-21 12:03:42
// MODIFIED: 2016-04-23 01:41:38
////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "class.h"
#include "class.r"
#include "../_utils/macros.h"

void * new ( const void * _class, ... )
{
    const struct Class * class = _class;
    void * mem = calloc( 1, class->size );
    
    //TODO: Replace with better error handling.
    assert ( mem );
    
    * ( const struct Class ** ) mem = class;
    
    // This handles and vars passed to the constructor.
    if ( class->__init__ )
    {
        va_list args;
        #ifdef DEBUG
            TRACE("Class Created", "__init__",class->__str__);
        #endif
        va_start( args, _class );                 // intialize '...'
	    mem = class->__init__( mem, &args );      // call constructor
        va_end( args );                           // clean
    }
    
    return mem;
}

void delete ( void * self )
{
    const struct Class ** parent = self;

    if ( self && * parent && ( * parent )->__delete__ ) {
        self = ( * parent )->__delete__(self);
        #ifdef DEBUG
            TRACE("Class Destroyed","delete(...)",class->__str__);
        #endif
    } 
    free(self);
}

void * cpy ( const void * self )
{
    const struct Class * const * parent = self;

    assert ( self && parent && ( * parent )->__copy__);
    
    #ifdef DEBUG
        TRACE("Class Copied","cpy(..)",class->__str__);
    #endif
    
    return ( * parent )->__copy__(self);
}

char * print ( const void * self )
{
    const struct Class * const * parent = self;
    char * pstring = malloc ( 255 * (sizeof(char)));
    sprintf(pstring,"%s",( * parent )->__str__(self));
    
    #ifdef DEBUG
        TRACE ( "Generated Print String: ", "print(...)", pstring );
    #endif
    
    return pstring;
}
