
#include "lbp.h"

/* ---------------------------------------------------------------- */
/*                                           Initialization helpers */
/* ---------------------------------------------------------------- */

int vl_lbp_get_dimension(VlLbp * self)
{
	return self->dimension ;
}
static void
_vl_lbp_init_uniform(VlLbp * self)
{
  int i, j ;

  /* one bin for constant patterns, 8*7 for 2-uniform, one for rest */
  self->dimension = 58 ;

  /* all but selected patterns map to 57 */
  for (i = 0 ; i < 256 ; ++i) {
    self->mapping[i] = 57 ;
  }

  /* uniform patterns map to 56 */
  self->mapping[0x00] = 56 ;
  self->mapping[0xff] = 56 ;

  /* now uniform pattenrs, in order */
  /* locations: 0:E, 1:SE, 2:S, ..., 7:NE */
  for (i = 0 ; i < 8 ; ++i) { /* string[i-1]=0, string[i]=1 */
    for (j = 1 ; j <= 7 ; ++j) { /* length of sequence of ones */
      /* string starting with j ones */
      int unsigned string = (1 << j) - 1 ;
      /* put i zeroes in front */
      string <<= i ;
      /* wrap around 8 bit boundaries */
      string = (string | (string >> 8)) & 0xff ;

      /* optionally transpose the pattern */
      if (self->transposed) {
        int unsigned original = string;
        int k ;
        /* flip the string left-right */
        string = 0 ;
        for (k = 0 ; k < 8 ; ++k) {
          string <<= 1 ;
          string |= original & 0x1  ;
          original >>= 1 ;
        }
        /* rotate 90 degrees */
        string <<= 3 ;
        string = (string | (string >> 8)) & 0xff ;
      }

      self->mapping[string] = i * 7 + (j-1) ;
    }
  }
}

/* ---------------------------------------------------------------- */

/** @brief Create a new LBP object
 ** @param type type of LBP features.
 ** @param transposed if @c true, then transpose each LBP pattern.
 ** @return new VlLbp object instance.
 **/

VlLbp *
vl_lbp_new(VlLbpMappingType type, bool transposed)
{
  VlLbp * self = (VlLbp *) malloc(sizeof(VlLbp)) ;
  self->transposed = transposed ;
  switch (type) {
    case VlLbpUniform: _vl_lbp_init_uniform(self) ; break ;
    default: exit(1) ;
  }
  return self ;
}

/** @brief Delete VlLbp object
 ** @param self object to delete.
 **/

void
vl_lbp_delete(VlLbp * self) {
  free(self) ;
}


/* ---------------------------------------------------------------- */

/** @brief Extract LBP features
 ** @param self LBP object.
 ** @param features buffer to write the features to.
 ** @param image image.
 ** @param width image width.
 ** @param height image height.
 ** @param cellSize size of the LBP cells.
 **
 ** @a features is a  @c numColumns x @c numRows x @c dimension where
 ** @c dimension is the dimension of a LBP feature obtained from ::vl_lbp_get_dimension,
 ** @c numColumns is equal to @c floor(width / cellSize), and similarly
 ** for @c numRows.
 **/

void vl_lbp_process (VlLbp * self,
                float * features,
                float * image, int width, int height,
                int cellSize)
{
  int cwidth = width / cellSize;
  int cheight = height / cellSize ;
  int cstride = cwidth * cheight ;
  int cdimension = vl_lbp_get_dimension(self) ;
  int x,y,cx,cy,k,bin ;

#define at(u,v) (*(image + width * (v) + (u)))
#define to(u,v,w) (*(features + cstride * (w) + cwidth * (v) + (u)))

  /* accumulate pixel-level measurements into cells */
  for (y = 1 ; y < (signed)height - 1 ; ++y) 
  {
    float wy1 = (y + 0.5f) / (float)cellSize - 0.5f ;
    int cy1 = (int) floor(wy1) ;
    int cy2 = cy1 + 1 ;
    float wy2 = wy1 - (float)cy1 ;
    wy1 = 1.0f - wy2 ;
    if (cy1 >= (signed)cheight) continue ;

    for (x = 1 ; x < (signed)width - 1; ++x) 
	{
      float wx1 = (x + 0.5f) / (float)cellSize - 0.5f ;
      int cx1 = (int) floor(wx1) ;
      int cx2 = cx1 + 1 ;
      float wx2 = wx1 - (float)cx1 ;
      wx1 = 1.0f - wx2 ;
      if (cx1 >= (signed)cwidth) continue ;

      {
        int unsigned bitString = 0 ;
        float center = at(x,y) ;
        if(at(x+1,y+0) > center) bitString |= 0x1 << 0; /*  E */
        if(at(x+1,y+1) > center) bitString |= 0x1 << 1; /* SE */
        if(at(x+0,y+1) > center) bitString |= 0x1 << 2; /* S  */
        if(at(x-1,y+1) > center) bitString |= 0x1 << 3; /* SW */
        if(at(x-1,y+0) > center) bitString |= 0x1 << 4; /*  W */
        if(at(x-1,y-1) > center) bitString |= 0x1 << 5; /* NW */
        if(at(x+0,y-1) > center) bitString |= 0x1 << 6; /* N  */
        if(at(x+1,y-1) > center) bitString |= 0x1 << 7; /* NE */
        bin = self->mapping[bitString] ;
      }

      if ((cx1 >= 0) & (cy1 >=0)) 
	  {
        to(cx1,cy1,bin) += wx1 * wy1;
      }
      if ((cx2 < (signed)cwidth)  & (cy1 >=0)) 
	  {
        to(cx2,cy1,bin) += wx2 * wy1 ;
      }
      if ((cx1 >= 0) & (cy2 < (signed)cheight)) 
	  {
        to(cx1,cy2,bin) += wx1 * wy2 ;
      }
      if ((cx2 < (signed)cwidth) & (cy2 < (signed)cheight)) 
	  {
        to(cx2,cy2,bin) += wx2 * wy2 ;
      }
    } /* x */
  } /* y */

  /* normalize cells */
  for (cy = 0 ; cy < (signed)cheight ; ++cy) 
  {
    for (cx = 0 ; cx < (signed)cwidth ; ++ cx) 
	{
      float norm = 0 ;
      for (k = 0 ; k < (signed)cdimension ; ++k) 
	  {
        norm += features[k * cstride] ;
      }
      norm = sqrt(norm) + 1e-10f; ;
      for (k = 0 ; k < (signed)cdimension ; ++k) 
	  {
        features[k * cstride] = sqrt(features[k * cstride]) / norm  ;
      }
      features += 1 ;
    }
  } /* next cell to normalize */
}

vectorf CalLBP(const Mat& im, int cellSize)
{
	
	int width = im.cols;
	int height = im.rows;
	int dims [3] ;

	/* get LBP object */
	VlLbp * lbp = vl_lbp_new (VlLbpUniform, true) ;
	/* get output buffer */
	dims[0] = height / cellSize ;
	dims[1] = width / cellSize ;
	dims[2] = vl_lbp_get_dimension(lbp) ;
	vectorf image(width*height,0);
	int num = 0;
	for (int i=0; i<height; i++)
	{
		for (int j=0; j<width; j++)
		{
			image[num++] = float (im.at<char>(i,j));
		}
	}
	vectorf fea(dims[0]*dims[1]*dims[2],0);
	vl_lbp_process(lbp, &fea[0], &image[0], height, width, cellSize) ;
	vl_lbp_delete(lbp) ;
	return fea;
}