/** @file lbp.h
 ** @brief Local Binary Patterns
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#ifndef H_LBP
#define H_LBP
#include "Public.h"


/** @brief Type of quantization for LBP features */
typedef enum _VlLbpMappingType
{
  VlLbpUniform     /**< Uniform patterns */
} VlLbpMappingType ;

/** @brief Local Binary Pattern extractor */
typedef struct VlLbp_
{
  int dimension ;
  int mapping [256] ;
  bool transposed ;
} VlLbp ;

VlLbp * vl_lbp_new(VlLbpMappingType type, bool transposed) ;
void vl_lbp_delete(VlLbp * self) ;
void vl_lbp_process (VlLbp * self,
                               float * features,
                               float * image, int width, int height,
                               int cellSize) ;
vectorf CalLBP(const Mat& im, int cellSize);

/** @brief Get the dimension of the LBP histograms
 ** @return dimension of the LBP histograms.
 ** The dimension depends on the type of quantization used.
 ** @see ::vl_lbp_new().
 **/

int vl_lbp_get_dimension(VlLbp * self);

#endif