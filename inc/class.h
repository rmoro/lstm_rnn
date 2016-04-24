////////////////////////////////////////////////////////////////////
// AUTHOR:   Robert Morouney <069001422>
// EMAIL:    robert@morouney.com
// FILE:     class.h
// CREATED:  2016-04-21 12:03:42
// MODIFIED: 2016-04-22 19:18:24
////////////////////////////////////////////////////////////////////

#ifndef NEW_H
#define NEW_H

void * new      ( const void * class, ... );
void delete     ( void * item );
void * cpy      ( const void * self );
char * print    ( const void * self );

#endif
