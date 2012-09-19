#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/saliency.c"
#else

/*======================================================================
 * File: saliency
 *
 * Description: Some ideas for a spatial saliency operator combining
 *  histograms and entropy on integer images.
 *
 * Created: May 16, 2012, 1:22AM
 *
 *
 * Author: Marco Scoffier // github@metm.org 
 *======================================================================*/

#include <luaT.h>
#include <TH.h>

#include <sys/param.h>
#include <sys/time.h>

/* maximum number of scales over which saliency is computed */
#define MAXSCALES 16

/*---------------------------------------------------------
 *
 * Computes the "integer image" of a tensor. In that tensor's default
 * type.  The int types should default to Long.
 * 
 *---------------------------------------------------------*/
static int libsaliency_(Main_intImage)(lua_State *L){
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);
  
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih  = src->size[0]; // #input channels
  long ir  = src->size[1]; // #input rows
  long ic  = src->size[2]; // #input cols
  long sch = src->stride[0]; 

  THTensor_(resize3d)(dst, ih, ir, ic);

  real *ii = THTensor_(data)(src);
  real *ri = THTensor_(data)(dst);

  real *rip,*rio;
  real row;
  long cc,xx,yy;
  
  // integer image is the sum of all the pixels from (0,0) top-left to (x,y)
    
  rio=ri;
  for(cc = 0; cc < ih; cc++){
    // Avoid branching.  Copy first elem first row.
    *ri = *ii; 
    rip = ri; // previous is to the left
    ri++; ii++;
    // do first col
    for(xx = 1; xx < ic; xx++){
      *ri = *ii + *rip;
      ri++ ; ii++; rip++;
    }
    // now rip tracks pointer in previous col
    rip = rio ;
    rio+=sch;
    for(yy = 1; yy < ir; yy++) {
      // copy first elem in row 
      row = 0; // row tracks to the left
      for(xx = 0; xx < ic; xx++) {
        row += *ii;
        *ri = row + *rip;
        ri++; ii++; rip++;
      }
    }
  }

  // cleanup
  THTensor_(free)(src);

  return 1;
}


/*---------------------------------------------------------
 *
 * An integral Histogram is like an integral image for each bin of the
 * histogram.
 * 
 * FIXME add flag for perchannel, perhaps we want to pass a vector to
 * keep things stable (or perhaps not).
 *---------------------------------------------------------*/
static int libsaliency_(Main_intHist)(lua_State *L){
  THTensor *dst = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  long nbins    = 16; 
  real minval   = 0;
  real maxval   = 0;
  real epsilon  = 1e-6; // larger than float machine precision
  char checkoverflow = 0;
  char perchannel    = 1;
  //struct timeval t1, t2;
  if (lua_isnumber(L,3)){
    nbins  = lua_tonumber(L,3);
  }
  if (lua_isnumber(L,4)){minval = lua_tonumber(L,4);}
  if (lua_isnumber(L,5)){maxval = lua_tonumber(L,5);}
  if ((minval == 0)&&(maxval == 0))
  {
    minval = THTensor_(minall)(src);
    maxval = THTensor_(maxall)(src);
  }
  // do we have another pass through the data to check the bins which
  // overflow
  if (((minval - THTensor_(minall)(src)) > epsilon) ||
      ((maxval - THTensor_(maxall)(src)) > epsilon)) {
    checkoverflow = 1;
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }
  // make sure we don't overflow
  minval -= epsilon;
  maxval += epsilon;
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  real *sd = THTensor_(data)(src);
  long ih  = src->size[0]; // #input channels
  long ir  = src->size[1]; // #input rows
  long ic  = src->size[2]; // #input cols

  long cc,i,xx,yy,bb;

  /*
   * First compute the histogram bins (in clone) */
  THTensor *clone = THTensor_(newWithSize3d)(ih,ir,ic);
  real *dorig = THTensor_(data)(clone); // pointer to uninitialized clone
  real *d     = dorig;
  long ne     = clone->stride[0];
  THTensor *srcPlane   = THTensor_(new)();
  THTensor *clonePlane = THTensor_(new)();
  // (3 rather than 5 op) histogram code 
  // FIXED !! do each channel independently
  for(cc = 0; cc < ih; cc++){
    // source channel
    THTensor_(select)(srcPlane, src, 0, cc);
    sd     = THTensor_(data)(srcPlane);
    // clone channel
    THTensor_(select)(clonePlane, clone, 0, cc); 
    ne     = THTensor_(nElement)(srcPlane);
    d      = THTensor_(data)(clonePlane);
    // compute per channel min and max
    if (perchannel > 0) {
      minval = THTensor_(minall)(srcPlane) - epsilon;
      maxval = THTensor_(maxall)(srcPlane) + epsilon;
    }
    THVector_(fill)(d,-minval,ne);
    THVector_(add)(d,sd,1,ne);
    THVector_(scale)(d,nbins/(maxval-minval),ne);
    if (checkoverflow > 0) {
      for (i = 0; i<ne; i++){
        d[i] = MIN(nbins-1,MAX(0,d[i]));
      }
    }
  }
  // reset
  d = dorig;

  THTensor_(resize4d)(dst, ih, ir, ic, nbins);
  real *ri  = THTensor_(data)(dst);
  
  real *rip,*rorig;
  /*
   * An integer image is the sum of all the pixels from (0,0) top-left
   * to (x,y).  Computing an integer image for the histogram is an
   * integer image per histogram bin.
   * 
   * This is slow.  :(
   * Time increases linearly with the number of histogram bins.
   * ~20ms for 4-8 bins > 670ms for 256 bins
   */
    
  rorig=ri;
  dorig=d;

  THTensor *row  = THTensor_(newWithSize1d)(nbins);
  real     *rowp = THTensor_(data)(row);
    
  /*
#pragma omp parallel for private(cc,xx,yy,bb,ri,rip,d) shared (ih,ir,ic,nbins,dorig,rorig)
  */
  for(cc = 0; cc < ih; cc++){
    d  = dorig + cc*clone->stride[0];
    ri = rorig + cc*dst->stride[0];
    // unroll fill w/ 0
    long uc = 4;
    for(bb = 0; bb <= nbins-uc; bb+=uc){
      ri[bb]    = 0;
      ri[bb+1]  = 0;
      ri[bb+2]  = 0;
      ri[bb+3]  = 0;
    }
    for(; bb<nbins;bb++){
      ri[bb] = 0;
    }
    // Avoid branching.  Bin the first elem of the first row.
    int bin = (int)floor(*d);
    ri[bin] = 1;
    rip = ri; // rip tracks previous entry (above)
    ri+=dst->stride[2]; 
  
    // do first col
    for(xx = 1; xx < ic; xx++){
      d++; bin = (int)floor(*d);
      // unroll copy hist bins 
      long uc = 4;
      for(bb = 0; bb <= nbins-uc; bb+=uc){
        ri[bb]    = rip[bb];
        ri[bb+1]  = rip[bb+1];
        ri[bb+2]  = rip[bb+2];
        ri[bb+3]  = rip[bb+3];
      }
      for(; bb<nbins;bb++){
        ri[bb]  = rip[bb];
      }
      // increment new bin
      ri[bin] += 1;
      ri+=dst->stride[2]; rip+=dst->stride[2];
    }
    
    // set rip to track pointer in previous col (to left)
    rip =  rorig + cc*dst->stride[0];
    for(yy = 1; yy < ir; yy++) {
      THTensor_(fill)(row,0); // row accumulates values 
      for(xx = 0; xx < ic; xx++) {
        d++; bin = (int)floor(*d);
        // unroll copy hist bins
        long uc = 4;
        rowp[bin] += 1;    // update into row storage
        for(bb = 0; bb <= nbins-uc; bb+=uc){
          ri[bb]    = rip[bb]   + rowp[bb]; 
          ri[bb+1]  = rip[bb+1] + rowp[bb+1];
          ri[bb+2]  = rip[bb+2] + rowp[bb+2];
          ri[bb+3]  = rip[bb+3] + rowp[bb+3];
        }
        for(; bb<nbins;bb++){
          ri[bb]  = rip[bb] + rowp[bb];
        }
        ri+=dst->stride[2]; rip+=dst->stride[2];
      }
    }
  } // loop cc
  // cleanup
  THTensor_(free)(row); 
  THTensor_(free)(clone);
  THTensor_(free)(src);
  
  return 1;
}

/*---------------------------------------------------------
 *
 * An integral Histogram is like an integral image for each bin of the
 * histogram.
 *
 * This function puts all color channels in the same histogram so that
 * after processing the output is 1 x H x W x nbins*nchan rather than
 * nchan x H x W x nbins
 * 
 *---------------------------------------------------------*/
static int libsaliency_(Main_intHistPack)(lua_State *L){
  THTensor *dst = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  long nbins    = 16; 
  real minval   = 0;
  real maxval   = 0;
  real epsilon  = 1e-6; // larger than float machine precision
  char checkoverflow = 0;
  char perchannel    = 1;
   //struct timeval t1, t2;
  if (lua_isnumber(L,3)){
    nbins  = lua_tonumber(L,3);
  }
  if (lua_isnumber(L,4)){minval = lua_tonumber(L,4);}
  if (lua_isnumber(L,5)){maxval = lua_tonumber(L,5);}
  if ((minval == 0)&&(maxval == 0))
  {
    minval = THTensor_(minall)(src);
    maxval = THTensor_(maxall)(src);
  }
  // do we have another pass through the data to check the bins which
  // overflow
  if (((minval - THTensor_(minall)(src)) > epsilon) ||
      ((maxval - THTensor_(maxall)(src)) > epsilon)) {
    checkoverflow = 1;
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }
  // make sure we don't overflow
  minval -= epsilon;
  maxval += epsilon;
  // make sure input is contiguous
  long cc,i,xx,yy,bb;
  src = THTensor_(newContiguous)(src);
  long ih  = src->size[0]; // #input channels
  long ir  = src->size[1]; // #input rows
  long ic  = src->size[2]; // #input cols
  real *sd = THTensor_(data)(src);

  /*
   * First compute the histogram bins (in clone) */
  THTensor *clone      = THTensor_(newWithSize3d)(ih,ir,ic);
  real *dorig          = THTensor_(data)(clone); 
  real *d              = dorig;
  long ne              = clone->stride[0];
  THTensor *srcPlane   = THTensor_(new)();
  THTensor *clonePlane = THTensor_(new)();
  // (3 rather than 5 op) histogram code 
  // FIXED !! do each channel independently
  // Ideally we would pass an array of nbins to allow for varying
  // number of bins per input channel.
  for(cc = 0; cc < ih; cc++){
    // source channel
    THTensor_(select)(srcPlane, src, 0, cc);
    sd     = THTensor_(data)(srcPlane);
    // clone channel
    THTensor_(select)(clonePlane, clone, 0, cc); 
    ne     = THTensor_(nElement)(srcPlane);
    d      = THTensor_(data)(clonePlane);
    // compute per channel min and max
    if (perchannel > 0) {
      minval = THTensor_(minall)(srcPlane) - epsilon;
      maxval = THTensor_(maxall)(srcPlane) + epsilon;
    }
    THVector_(fill)(d,-minval,ne);
    THVector_(add)(d,sd,1,ne);
    THVector_(scale)(d,nbins/(maxval-minval),ne);
    if (checkoverflow > 0) {
      for (i = 0; i<ne; i++){
        d[i] = MIN(nbins-1,MAX(0,d[i]));
      }
    }
  }
  // reset
  d = dorig;
  
  // Fix we are packing the colors together into a single
  // histogram
  THTensor_(resize4d)(dst, 1 , ir, ic, ih*nbins);
  real *ri  = THTensor_(data)(dst);
  real *rip,*rorig;

  /*
   * An integer image is the sum of all the pixels from (0,0) top-left
   * to (x,y).  Computing an integer image for the histogram is an
   * integer image per histogram bin.
   * 
   * This is slow.  :(
   * Time increases linearly with the number of histogram bins.
   * ~20ms for 4-8 bins > 670ms for 256 bins
   */
    
  rorig=ri;

  THTensor *row  = THTensor_(newWithSize1d)(dst->stride[2]);
  real     *rowp = THTensor_(data)(row);

  /* now we collect data from all channels in a single pass
   * so store a pointer for each channel */
  real *chanp[ih];
  chanp[0] = THTensor_(data)(clone);
  for(cc = 1; cc < ih; cc++){
    chanp[cc] = chanp[0] + cc*clone->stride[0];
  }
  
  /*
#pragma omp parallel for private(cc,xx,yy,bb,ri,rip,d) shared (ih,ir,ic,nbins,dorig,rorig)
  */
  // result 1 x nrow x ncol x nchannels * nbins
  ri = rorig;
  // unroll fill w/ 0
  long uc = 4;
  for(bb = 0; bb <= dst->stride[2]-uc; bb+=uc){
    ri[bb]    = 0;
    ri[bb+1]  = 0;
    ri[bb+2]  = 0;
    ri[bb+3]  = 0;
  }
  for(; bb<dst->stride[2];bb++){
    ri[bb] = 0;
  }
  int bin = 0;
  // Avoid branching.  Bin the first elem of the first row.
  // get all the channels
  for(cc = 0; cc < ih; cc++){
    bin = (int)floor(*(chanp[cc]));
    bin += cc*nbins;
    ri[bin] = 1;
  }
  rip  = ri; // rip tracks previous entry (above) for next row
  ri  += dst->stride[2]; 
  
  // do first col (starting at second element)
  for(xx = 1; xx < ic; xx++){
    // unroll copy hist bins 
    long uc = 4;
    for(bb = 0; bb <= dst->stride[2]-uc; bb+=uc){
      ri[bb]    = rip[bb];
      ri[bb+1]  = rip[bb+1];
      ri[bb+2]  = rip[bb+2];
      ri[bb+3]  = rip[bb+3];
    }
    for(; bb<dst->stride[2];bb++){
        ri[bb]  = rip[bb];
    }
    // increment new bin
    for(cc = 0; cc < ih; cc++){
      (chanp[cc])++;
      bin = (int)floor(*(chanp[cc]));
      bin += cc*nbins;
      ri[bin] += 1;
    }
    ri+=dst->stride[2];
    rip+=dst->stride[2];
  }
  
  // reset rip to track pointer in previous col (to left)
  rip = rorig;
  for(yy = 1; yy < ir; yy++) {
    THTensor_(fill)(row,0); // row accumulates values 
    for(xx = 0; xx < ic; xx++) {
      // increment new bin
      for(cc = 0; cc < ih; cc++){
        (chanp[cc])++;
        bin = (int)floor(*(chanp[cc]));
        bin += cc*nbins;
        rowp[bin] += 1; // update into row storage
      }
      // unroll copy hist bins
      long uc = 4;
      for(bb = 0; bb <= dst->stride[2]-uc; bb+=uc){
        ri[bb]    = rip[bb]   + rowp[bb]; 
        ri[bb+1]  = rip[bb+1] + rowp[bb+1];
        ri[bb+2]  = rip[bb+2] + rowp[bb+2];
        ri[bb+3]  = rip[bb+3] + rowp[bb+3];
      }
      for(; bb<dst->stride[2];bb++){
        ri[bb]  = rip[bb] + rowp[bb];
      }
      ri  += dst->stride[2];
      rip += dst->stride[2];
    }
  }

  // cleanup
  THTensor_(free)(row);
  THTensor_(free)(clone);
  THTensor_(free)(src);
  THTensor_(free)(srcPlane);
  THTensor_(free)(clonePlane);
  return 1;
}

/*---------------------------------------------------------
 *
 * An integral Histogram is like an integral image for each bin of the
 * histogram. uses Long and binary math for speed
 * 
 *---------------------------------------------------------*/
static int libsaliency_(Main_intHistLong)(lua_State *L){
  THLongTensor *dst = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  long nbins    = 16; 
  real minval   = 0;
  real maxval   = 0;
  real epsilon  = 1e-6; // larger than float machine precision
  char checkoverflow = 0;
  //struct timeval t1, t2;
  if (lua_isnumber(L,3)){
    nbins  = lua_tonumber(L,3);
  }
  if (lua_isnumber(L,4)){minval = lua_tonumber(L,4);}
  if (lua_isnumber(L,5)){maxval = lua_tonumber(L,5);}
  if ((minval == 0)&&(maxval == 0))
  {
    minval = THTensor_(minall)(src);
    maxval = THTensor_(maxall)(src);
  }
  // do we have another pass through the data to check the bins which
  // overflow
  if (((minval - THTensor_(minall)(src)) > epsilon) ||
      ((maxval - THTensor_(maxall)(src)) > epsilon)) {
    checkoverflow = 1;
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }
  // make sure we don't overflow
  minval -= epsilon;
  maxval += epsilon;
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  real *sd = THTensor_(data)(src);
  long ih  = src->size[0]; // #input channels
  long ir  = src->size[1]; // #input rows
  long ic  = src->size[2]; // #input cols

  /*
   * First compute the histogram bins (in clone) */
  THTensor *clone = THTensor_(newWithSize3d)(ih,ir,ic);
  real *d = THTensor_(data)(clone);
  long ne = THTensor_(nElement)(clone);
  int i;
  // (3 rather than 5 op) histogram code
  THVector_(fill)(d,-minval,ne);
  THVector_(add)(d,sd,1,ne);
  THVector_(scale)(d,nbins/(maxval-minval),ne);
  if (checkoverflow > 0) {
    for (i = 0; i<ne; i++){
      d[i] = MIN(nbins-1,MAX(0,d[i]));
    }
  }

  THLongTensor_resize4d(dst, ih, ir, ic, nbins);
  long *ri  = THLongTensor_data(dst);
  long *rorig,*rip;
  real *dorig;
  long cc,xx,yy,bb;
  long bin = -1;
  /*
   * An integer image is the sum of all the pixels from (0,0) top-left
   * to (x,y).  Computing an integer image for the histogram is an
   * integer image per histogram bin.
   * 
   * This is slow.  :(
   * Time increases linearly with the number of histogram bins.
   * ~20ms for 4-8 bins > 670ms for 256 bins
   */
    
  rorig=ri;
  dorig=d;

  THLongTensor *row  = THLongTensor_newWithSize1d(nbins);
  long         *rowp = THLongTensor_data(row);
  /*
#pragma omp parallel for private(cc,xx,yy,bb,ri,rip,d) shared (ih,ir,ic,nbins,dorig,rorig)
  */
  for(cc = 0; cc < ih; cc++){
    d  = dorig + cc*clone->stride[0];
    ri = rorig + cc*dst->stride[0];
    // unroll fill w/ 0
    long uc = 4;
    for(bb = 0; bb <= nbins-uc; bb+=uc){
      ri[0]  = 0;
      ri[1]  = 0;
      ri[2]  = 0;
      ri[3]  = 0;
    }
    for(; bb<nbins;bb++){
      ri[bb] = 0;
    }
    // Avoid branching.  Bin the first elem of the first row.
    bin = (int)floor(*d);
    ri[bin] = 1;
    rip = ri; // rip tracks previous entry (above)
    ri+=dst->stride[2]; 
  
    // do first col
    for(xx = 1; xx < ic; xx++){
      d++; bin = (long)floor(*d);
      // unroll copy hist bins 
      long uc = 4;
      for(bb = 0; bb <= nbins-uc; bb+=uc){
        ri[0]  = rip[0];
        ri[1]  = rip[1];
        ri[2]  = rip[2];
        ri[3]  = rip[3];
      }
      for(; bb<nbins;bb++){
        ri[bb]  = rip[bb];
      }
      // increment new bin
      ri[bin] += 1;
      ri+=dst->stride[2]; rip+=dst->stride[2];
    }
    
    // set rip to track pointer in previous col (to left)
    rip =  rorig + cc*dst->stride[0];
    for(yy = 1; yy < ir; yy++) {
      THLongTensor_fill(row,0); // row accumulates values 
      for(xx = 0; xx < ic; xx++) {
        d++; bin = (long)floor(*d);
        // unroll copy hist bins
        long uc = 4;
        rowp[bin] += 1;    // update into row storage
        for(bb = 0; bb <= nbins-uc; bb+=uc){
          ri[bb]    = rip[bb]   + rowp[bb]; 
          ri[bb+1]  = rip[bb+1] + rowp[bb+1];
          ri[bb+2]  = rip[bb+2] + rowp[bb+2];
          ri[bb+3]  = rip[bb+3] + rowp[bb+3];
        }
        for(; bb<nbins;bb++){
          ri[bb]  = rip[bb] + rowp[bb];
        }
        ri+=dst->stride[2]; rip+=dst->stride[2];
      }
    }
  } // loop cc

  // cleanup
  THLongTensor_free(row); 
  THTensor_(free)(clone);
  THTensor_(free)(src);
  
  return 1;
}

/*
 * intAvg takes an integer image or integer hist as input.
 * 
 * The code basically operates as a convolution which takes the sum of
 * each window. But because of the integer image the time is constant
 * for any kernel size.
 *
 * //convolution modeled off THTensorConv.c
 */
static int libsaliency_(Main_intAvg)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  /* Get Args. */
  /* // Window (kernel) in which we compute the hist */
  long kr             = lua_tonumber(L, 3); // #kernel rows
  long kc             = lua_tonumber(L, 4); // #kernel cols
  real n              = 1/(real)(kr * kc);  // normalization
  long sr             = 1;                  // #step rows
  long sc             = 1;                  // #step cols

  if (lua_isnumber(L, 5)){sr = lua_tonumber(L,5);}
  if (lua_isnumber(L, 6)){sc = lua_tonumber(L,6);}

  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input channels
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long ndims = THTensor_(nDimension)(src);
  long ib  = 1;
  if (ndims >3){
    ib = src->size[3]; // #input bins
  }
  long ssh = src->stride[0]; // #input channels
  long isr = src->stride[1]; // #input rows
  long isc = src->stride[2]; // #input cols
  // pointers to data (four corners of a box)
  real *ic1 = THTensor_(data)(src);
  real *ic2, *ic3, *ic4;
  ic2 = ic1 + (kc-1) * isc; // over kc cols
  ic3 = ic1 + (kr-1) * isr; // down kr rows
  ic4 = ic3 + (kc-1) * isc; // over kc cols
  real *ir1, *ir2, *ir3, *ir4;
  ir1 = ic1; ir2 = ic2; ir3 = ic3; ir4 = ic4;
  real *ih1, *ih2, *ih3, *ih4;
  ih1 = ic1; ih2 = ic2; ih3 = ic3; ih4 = ic4;
  
  long or = (ir - kr)/sr + 1;
  long oc = (ic - kc)/sc + 1;
  
  long hh, rr, cc, bb;
  long ssr = sr*isr;
  long ssc = sc*isc; // inner most stride

  if (ib == 1){ // avg setting
    THTensor_(resize3d)(dst, ih, or, oc);
    // #pragma omp parallel
    for(hh = 0; hh < ih; hh++) {
      real *rs = THTensor_(data)(dst) + hh*dst->stride[0];
      // first element
      *rs = *ic4 * n;
      rs++;ic4+=ssc;
      //first row
      for(cc = 1; cc < oc; cc++){
        *rs = (*ic4 - *ic3)*n;
        rs++; ic4+=ssc; ic3+=ssc;
      }
      // step row
      ir3+=ssr; ir4+=ssr;
      ic3=ir3; ic4=ir4;
      for(rr = 1; rr < or; rr++) {
        // first element in row
        *rs = (*ic4 - *ic2) * n;
        rs++; ic4+=ssc; ic2+=ssc;
        // rest of row
        for(cc = 1; cc < oc; cc++) {
          *rs = (*ic4 - *ic2 - *ic3 + *ic1) * n;
          rs++;
          ic1+=ssc; ic2+=ssc; ic3+=ssc; ic4+=ssc; 
        }
        // step row
        ir1+=ssr; ir2+=ssr; ir3+=ssr; ir4+=ssr;
        ic1=ir1; ic2=ir2; ic3=ir3; ic4=ir4;
      }
      // step channel
      ih1+=ssh; ih2+=ssh; ih3+=ssh; ih4+=ssh;
      ir1=ih1; ir2=ih2; ir3=ih3; ir4=ih4;
      ic1=ih1; ic2=ih2; ic3=ih3; ic4=ih4;
    }
  } else { // histogram setting
    THTensor_(resize4d)(dst,ih,or,oc,ib);
/* #pragma omp parallel for private(hh,rr,cc,bb,rs,hstep, \ */
/*                                  ic1,ic2,ic3,ic4,      \ */
/*                                  ir1,ir2,ir3,ir4,      \ */
/*                                  ih1,ih2,ih3,ih4)      \ */
/*                          shared (n,oc,or,dst) */
    for(hh = 0; hh < ih; hh++) {
      // result
      real *rs = THTensor_(data)(dst)+hh*dst->stride[0]; 
      long uc = 4;
      // first element
      // unroll copy -- *rs = *ic4 * n; 
      for(bb = 0; bb <= ib-uc; bb+=uc){
        rs[bb]   = ic4[bb]  *n;
        rs[bb+1] = ic4[bb+1]*n;
        rs[bb+2] = ic4[bb+2]*n;
        rs[bb+3] = ic4[bb+3]*n;
      }
      for(; bb<ib;bb++){
        rs[bb]   = ic4[bb]  *n;
      }
      rs+=dst->stride[2]; ic4+=ssc;
      //first row -- *rs = (*ic4 - *ic3)*n;
      for(cc = 1; cc < oc; cc++){
        for(bb = 0; bb <= ib-uc; bb+=uc){
          rs[bb]   = (ic4[bb]   - ic3[bb])  *n;
          rs[bb+1] = (ic4[bb+1] - ic3[bb+1])*n;
          rs[bb+2] = (ic4[bb+2] - ic3[bb+2])*n;
          rs[bb+3] = (ic4[bb+3] - ic3[bb+3])*n; 
        }
        for(; bb<ib;bb++){
          rs[bb]   = (ic4[bb]   - ic3[bb])  *n;
        } 
        rs+=dst->stride[2]; ic4+=ssc; ic3+=ssc;
      }
      // step row
      ir3+=ssr; ir4+=ssr;
      ic3=ir3; ic4=ir4;

      for(rr = 1; rr < or; rr++) {
        // first element in row -- *rs = (*ic4 - *ic2) * n;
        for(bb = 0; bb <= ib-uc; bb+=uc){
          rs[bb]   = (ic4[bb]   - ic2[bb])  *n;
          rs[bb+1] = (ic4[bb+1] - ic2[bb+1])*n;
          rs[bb+2] = (ic4[bb+2] - ic2[bb+2])*n;
          rs[bb+3] = (ic4[bb+3] - ic2[bb+3])*n; 
        }
        for(; bb<ib;bb++){
          rs[bb]   = (ic4[bb]   - ic2[bb])  *n;
        } 
        rs+=dst->stride[2]; ic4+=ssc; ic2+=ssc;
        // rest of row  -- *rs = (*ic4 - *ic2 - *ic3 + *ic1) * n; 
        for(cc = 1; cc < oc; cc++) {
          // unroll (faster than using TH_Vector Ops
          for(bb = 0; bb <= ib-uc; bb+=uc){
            rs[bb]  =(ic4[bb]  -ic2[bb]  -ic3[bb]  +ic1[bb]  )*n;
            rs[bb+1]=(ic4[bb+1]-ic2[bb+1]-ic3[bb+1]+ic1[bb+1])*n;
            rs[bb+2]=(ic4[bb+2]-ic2[bb+2]-ic3[bb+2]+ic1[bb+2])*n;
            rs[bb+3]=(ic4[bb+3]-ic2[bb+3]-ic3[bb+3]+ic1[bb+3])*n;
          }
          for(; bb<ib;bb++){
            rs[bb]  = (ic4[bb]-ic2[bb]-ic3[bb]+ic1[bb])*n;
          }
          rs+=dst->stride[2];
          ic1+=ssc; ic2+=ssc; ic3+=ssc; ic4+=ssc;
        }
        // step row
        ir1+=ssr; ir2+=ssr; ir3+=ssr; ir4+=ssr;
        ic1=ir1; ic2=ir2; ic3=ir3; ic4=ir4;
      }
      // step channel
      ih1+=ssh; ih2+=ssh; ih3+=ssh; ih4+=ssh;
      ir1=ih1; ir2=ih2; ir3=ih3; ir4=ih4;
      ic1=ih1; ic2=ih2; ic3=ih3; ic4=ih4; 
    }
  }


  // cleanup
  THTensor_(free)(src);

  return 0;
}

/*
 * spatialMax: returns max of each pixel in each channel of the
 * spatialHist structure. Flattens the distribution replacing
 * the histogram with height of its largest bin.
 * 
 */
static int libsaliency_(Main_spatialMax)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  THArgCheck(src->nDimension == 4, 1,
             "input is 4D hist tensor (chan,row,col,nbins)");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input channels
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long ib    = src->size[3]; // #bins in hist
  
  THTensor_(resize2d)(dst, ir, ic);
  THTensor_(fill)(dst,0);
  
  real *ss = THTensor_(data)(src);
  real *dd = THTensor_(data)(dst);
  
  long hh, rr, cc, bb;
  // #pragma omp parallel
  for(hh = 0; hh < ih; hh++) {
    real theMax;
    for(rr = 0; rr < ir; rr++) {
      for(cc = 0; cc < ic; cc++) {
        theMax = *ss; ss++;
        for (bb = 1; bb < ib; bb++){
          if (*ss > theMax){ theMax = *ss; }
          ss++;
        }
        // max over all channels
        if(*dd < theMax){ *dd = theMax; };
        dd++;
      }
    }
    dd =  THTensor_(data)(dst);
  }
  // cleanup
  THTensor_(free)(src);

  return 0;  
}

/*
 * spatialOneOverMax: sums 1/max of each channel of the spatialHist structure
 * (simple approx to inverse entropy)
 * 
 */
static int libsaliency_(Main_spatialOneOverMax)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  THArgCheck(src->nDimension == 4, 1,
             "input is 4D hist tensor (chan,row,col,nbins)");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input channels
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long ib    = src->size[3]; // #bins in hist
  
  THTensor_(resize2d)(dst, ir, ic);
  THTensor_(fill)(dst,0);
  
  real *ss = THTensor_(data)(src);
  real *dd = THTensor_(data)(dst);
  
  long hh, rr, cc, bb;
  // #pragma omp parallel
  for(hh = 0; hh < ih; hh++) {
    real theMax;
    for(rr = 0; rr < ir; rr++) {
      for(cc = 0; cc < ic; cc++) {
        theMax = *ss; ss++;
        for (bb = 1; bb < ib; bb++){
          if (*ss > theMax){ theMax = *ss; }
          ss++;
        }
        // sum the channels
        *dd += 1/theMax;
        dd++;
      } // for cc
    } // for rr
    dd =  THTensor_(data)(dst);
  }
  // cleanup
  THTensor_(free)(src);

  return 0;  
}

/*
 * spatialMax: sums mean/max of each channel of the spatialHist structure
 * (simple approx to inverse entropy)
 * 
 */
static int libsaliency_(Main_spatialMeanOverMax)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  THArgCheck(src->nDimension == 4, 1,
             "input is 4D hist tensor (chan,row,col,nbins)");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input channels
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long ib    = src->size[3]; // #bins in hist
  
  THTensor_(resize2d)(dst, ir, ic);
  THTensor_(fill)(dst,0);
  
  real *ss = THTensor_(data)(src);
  real *dd = THTensor_(data)(dst);
  
  long hh, rr, cc, bb;
  // #pragma omp parallel
  for(hh = 0; hh < ih; hh++) {
    real theMax;
    real theSum;
    for(rr = 0; rr < ir; rr++) {
      for(cc = 0; cc < ic; cc++) {
        theMax = ss[0]; 
        theSum = ss[0];
        for (bb = 1; bb < ib; bb++){
          if (ss[bb] > theMax){ theMax = ss[bb]; }
          theSum += ss[bb];
        }
        *dd += theSum/(theMax*ib); 
        
        ss+=ib; 
        dd++;
      } // for cc
    } // for rr
    // reset
    dd =  THTensor_(data)(dst);
  }
  // cleanup
  THTensor_(free)(src);

  return 0;  
}

/*
 * scaleSaliency: computes saliency over scales which as in Kadir 2004
 * is the sum of absolute change in entropy between scales.
 */
static int libsaliency_(Main_scaleSaliency)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  THArgCheck(src->nDimension == 3, 1,
             "input is 3D scales tensor (scales,row,col)");
  THArgCheck(src->size[0] <= MAXSCALES, 1,
             "more than maximum number of scales");
  THArgCheck(src->size[0] >= 1, 1,
             "can't compute saliency of one scale");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input scales
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long hs    = src->stride[0];
  
  THTensor_(resize2d)(dst, ir, ic);
  THTensor_(fill)(dst,0);
  
  real *sd = THTensor_(data)(src);
  real *dd = THTensor_(data)(dst);
  real *sc[MAXSCALES];
  
  long ss, rr, cc;
  // this data struct is organized <scales> <row> <col> so we keep a
  // pointer to the head of each scale
  for(ss = 0; ss < ih; ss++) {
    sc[ss] = sd+hs*ss;
  }
  // now loop once through the rows and columns
  // #pragma omp parallel
  for(rr = 0; rr < ir; rr++) {
    for(cc = 0; cc < ic; cc++) {
      for(ss = 0; ss < ih-1; ss++) {
        *dd += fabs(*sc[ss+1] - *sc[ss]);
        sc[ss]++;
      }
      sc[ih-1]++; // increment the last one
      dd++;
    }
  }

  // cleanup
  THTensor_(free)(src);

  return 0;  
}

/*
 * spatialEnt: computes entropy of each PDF in spatialHist structure
 */
static int libsaliency_(Main_spatialEnt)(lua_State *L) {
  THTensor *dst =
    luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src =
    luaT_checkudata(L, 2, torch_Tensor);

  THArgCheck(src->nDimension == 4, 1,
             "input is 4D hist tensor (chan,row,col,nbins)");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  long ih    = src->size[0]; // #input channels
  long ir    = src->size[1]; // #input rows
  long ic    = src->size[2]; // #input cols
  long ib    = src->size[3]; // #bins in hist
  
  THTensor_(resize2d)(dst, ir, ic);
  THTensor_(fill)(dst,0);
  
  real *ss = THTensor_(data)(src);
  real *dd = THTensor_(data)(dst);
  
  long hh, rr, cc, bb;
  //#pragma omp parallel
  for(hh = 0; hh < ih; hh++) {
    real sum;
    for(rr = 0; rr < ir; rr++) {
      for(cc = 0; cc < ic; cc++) {
        sum = 0;
        for (bb = 0; bb < ib; bb++){
          if (*ss > 0){
            sum+=*ss * log(*ss);
          }
          ss++;
        }
        // sum the channels
        *dd -= sum;
        dd++;
      }
    }
    // reset to the origin
    dd =  THTensor_(data)(dst);
  }
  // cleanup
  THTensor_(free)(src);

  return 0;  
}

/*
 * nonMaximalSuppression: which sorts and returns best matches. Uses
 * Michael's trick to only search on horizonal and vertical line
 * around point.
 */
static int libsaliency_(Main_fastNMS)(lua_State *L) {
  THTensor *mat    = luaT_checkudata(L, 1, torch_Tensor);
  long windowsize  = 5;
  if (lua_isnumber(L,2)){
    windowsize  = lua_tonumber(L, 2);
  }
  long hw          = floor(windowsize*0.5);
  long npts        = -1;
  if (lua_isnumber(L,3)){
    npts = lua_tonumber(L,3);
  }
  // if not buf, bufv ...
  THTensor *buf;
  if (luaT_isudata(L,4,torch_Tensor)){
    buf = luaT_toudata(L, 4, torch_Tensor);
  } else {
    buf = THTensor_(new)();
  }
  THTensor *bufv;
  if (luaT_isudata(L,5,torch_Tensor)){
    bufv = luaT_toudata(L, 5, torch_Tensor);
  } else {
    bufv = THTensor_(new)();
  }
  
  THTensor *matr   = THTensor_(new)();
  THTensor_(unfold)(matr,mat,0,windowsize,1);
  THTensor *matc   = THTensor_(new)();
  THTensor_(unfold)(matc,mat,1,windowsize,1);

  THTensor *rv     = THTensor_(new)();
  THTensor *cv     = THTensor_(new)();
  THLongTensor *ri = THLongTensor_new();
  THLongTensor *ci = THLongTensor_new();
  THTensor_(max)(rv,ri,matr,2);
  THTensor_(max)(cv,ci,matc,2);

  long outr = rv->size[0];
  long outc = cv->size[1];
  THTensor_(resize2d)(buf,outr*outc,2);
  THTensor_(resize1d)(bufv,outr*outc);
  real * bufvd = THTensor_(data)(bufv);


  /*
  * Check equality of the horzontal window and vertical window.
  */
  long i,j,k,offr,offc;
  k = 0;
  real matrv,matcv,matv;
  for (i=0; i < outr; i++){
    for (j=0; j < outc; j++){
      offr = i + hw;
      offc = j + hw;
      matv  = THTensor_(get2d)(mat, offr, offc); 
      matrv = THTensor_(get3d)(rv, i, offc,0);
      matcv = THTensor_(get3d)(cv, offr, j,0);
      if ((matrv == matcv) && (matrv > 0) && (matrv == matv)) { 
        THTensor_(set2d)(buf,k,0,offc+1); 
        THTensor_(set2d)(buf,k,1,offr+1); 
        bufvd[k] = matv;
        k += 1;
      }
    }
  }
  /* 
  *   Sort the points found.
  */
  if (k>0) {
    // don't sort on more points than we found
    THTensor_(narrow)(buf,NULL,0,0,k);
    THTensor_(narrow)(bufv,NULL,0,0,k);
    THTensor *outv      = THTensor_(new)();
    THLongTensor *sorti = THLongTensor_new();
    THTensor_(sort)(outv,sorti,bufv,0,1);
    long *sd            = THLongTensor_data(sorti);
    // make sure we don't return more than asked for
    if ((npts>0)&&(npts<k)) { k = npts; }
    THTensor *outxy     = THTensor_(newWithSize2d)(k,2);
    real * outd         = THTensor_(data)(outxy);
    real * bufd         = THTensor_(data)(buf);
    // copy the locations of the highest scoring points
    for (i=0;i<k;i++){
      outd[(long)(i*outxy->size[1])]   =
        bufd[(long)(sd[i]*buf->size[1])];
      outd[(long)(i*outxy->size[1]+1)] =
        bufd[(long)(sd[i]*buf->size[1]+1)];
    }
    // return a tensor
    luaT_pushudata(L, outxy, torch_Tensor);
    luaT_pushudata(L, outv, torch_Tensor);
    lua_pushnumber(L,k);
    THLongTensor_free(sorti);
    THTensor_(free)(buf);
    THTensor_(free)(bufv);
  } else {
    // return a tensor
    luaT_pushudata(L, buf,  torch_Tensor);
    luaT_pushudata(L, bufv, torch_Tensor);
    lua_pushnumber(L,0);
  }
  // cleanup
  THTensor_(free)(matr);
  THTensor_(free)(matc);
  THTensor_(free)(rv);
  THTensor_(free)(cv);
  THLongTensor_free(ri);
  THLongTensor_free(ci);

  return 3;  
}

/*
 * nonMaximalSuppression: which sorts and returns best matches. Uses
 * Michael's trick to only search on horizonal and vertical line
 * around point.
 */
static int libsaliency_(Main_newNMS)(lua_State *L) {
  THTensor *src    = luaT_checkudata(L, 1, torch_Tensor);
  // check that src is 2d
  THArgCheck(src->nDimension == 2, 1, "input must be 2D");
  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  
  long wxh = src->size[0]*src->size[1];
  long windowsize  = 5;
  if (lua_isnumber(L,2)){
    windowsize  = lua_tonumber(L, 2);
  }
  long hw          = floor(windowsize*0.5);
  long outh        = src->size[0] - windowsize + 1;
  long outw        = src->size[1] - windowsize + 1;
  long npts        = -1;
  if (lua_isnumber(L,3)){
    npts = lua_tonumber(L,3);
  }
  // if not buf, bufv ...
  THTensor *buf;
  if (luaT_isudata(L,4,torch_Tensor)){
    buf = luaT_toudata(L, 4, torch_Tensor);
    THTensor_(resize2d)(buf,wxh,2);

  } else {
    buf = THTensor_(newWithSize2d)(wxh,2);
  }
  real * bufd = THTensor_(data)(buf);

  THTensor *bufv;
  if (luaT_isudata(L,5,torch_Tensor)){
    bufv = luaT_toudata(L, 5, torch_Tensor);
    THTensor_(resize1d)(bufv,wxh);
  } else {
    bufv = THTensor_(newWithSize1d)(wxh);
  }
  real * bufvd = THTensor_(data)(bufv);
  
  THTensor *tmpC = THTensor_(newWithSize2d)(outh,src->size[1]);
  // for input
  real * sd;
  real * ssd;
  // for output
  real * cd;
  
  /*
   * Take max over columns (non-continguous dimension), put in tmpC
   */
  long i,j,k;
  long ncwindowsize = windowsize * src->stride[0];
  real maxv;
  for (i=0; i < src->size[1]; i++){
    // set data pointers to top of a column;
    cd = THTensor_(data)(tmpC) + i;
    sd = THTensor_(data)(src) + i;
    // reset max
    maxv = -HUGE_VAL;
    // do first windowsize (from 0 -> ws-1)
    for (k=0; k < ncwindowsize; k+=src->stride[0] ){
      maxv = THMax(maxv,sd[k]);
    }
    *cd = maxv; cd+=src->stride[0];
    // j is just a counter, doesn't index anything
    for (j=windowsize; j < src->size[0]; j++){
      // test if we need to recompute the max add last element (max)
      if (sd[0] == maxv){
        // reset max
        maxv = -HUGE_VAL;
        for (k=src->stride[0]; k <= ncwindowsize; k+=src->stride[0] ){
          maxv = THMax(maxv,sd[k]);
        }
      } else {
        maxv = THMax(maxv,sd[ncwindowsize]);
      }
      *cd = maxv;
      sd+=src->stride[0]; cd+=src->stride[0];
    }
  }

  /*
   * Take max in each row of the column max, if this value is equal to
   * the input we have a maximal point and we store it in buf.
   */
  long fpts = 0;
  int off   = hw + 1;
  // (s)sd is used for comparisons we choose the middle point of each
  // sliding window for the comparison.
  ssd = THTensor_(data)(src) + hw*src->stride[0] + hw;
  for (i=0; i < outh; i++){
    // reset source data pointer
    sd = ssd + i*src->stride[0];
    // reset column data pointer
    cd = THTensor_(data)(tmpC) + i*src->stride[0];
    
    // reset max
    maxv = -HUGE_VAL;
    // do first windowsize (from 0 -> ws-1)
    for (k=0; k < windowsize; k++ ){
      maxv = THMax(maxv,cd[k]);
    }
    if (maxv == *sd) {
      // we have a point, opencv order
      *bufd = off; bufd++;
      *bufd = i + off; bufd++;
      *bufvd = maxv; bufvd++;
      fpts++;
    }
    sd++;
    for (j=1; j < outw; j++){
      // test if we need to recompute the max add last element (max)
      if (cd[0] == maxv){
        // reset max
        maxv = -HUGE_VAL;
        // compute max
        for (k=1; k <= windowsize; k++ ){
          maxv = THMax(maxv,cd[k]);
        }
      } else {
        maxv = THMax(maxv,cd[windowsize]);
      }
      if (maxv == *sd) {
        // we have a point, opencv order
        *bufd = j + off; bufd++;
        *bufd = i + off; bufd++;
        *bufvd = maxv; bufvd++;
        fpts++;
      }
      // everything is contiguous
      sd++; cd++;
    }
  }

  /* 
   *   Sort the points found.
   */
  if (fpts>0) {
    // don't sort on more points than we found
    THTensor_(narrow)(buf,NULL,0,0,fpts);
    THTensor_(narrow)(bufv,NULL,0,0,fpts);
    THTensor *outv      = THTensor_(new)();
    THLongTensor *sorti = THLongTensor_new();
    THTensor_(sort)(outv,sorti,bufv,0,1);
    long *sd            = THLongTensor_data(sorti);
    // make sure we don't return more than asked for
    if ((npts>0)&&(npts<fpts)) { fpts = npts; }
    THTensor *outxy     = THTensor_(newWithSize2d)(fpts,2);
    real * outd         = THTensor_(data)(outxy);
    real * bufd         = THTensor_(data)(buf);
    // copy the locations of the highest scoring points
    for (i=0;i<fpts;i++){
      outd[(long)(i*outxy->size[1])]   =
        bufd[(long)(sd[i]*buf->size[1])];
      outd[(long)(i*outxy->size[1]+1)] =
        bufd[(long)(sd[i]*buf->size[1]+1)];
    }
    // return a tensor
    luaT_pushudata(L, outxy, torch_Tensor);
    luaT_pushudata(L, outv, torch_Tensor);
    lua_pushnumber(L,fpts);
    THLongTensor_free(sorti);
  } else {
    // return a tensor
    luaT_pushudata(L, buf,  torch_Tensor);
    luaT_pushudata(L, bufv, torch_Tensor);
    lua_pushnumber(L,0);
  }
    
  // cleanup
  THTensor_(free)(src);
  THTensor_(free)(tmpC);
  return 3;  
}


//============================================================
// Register functions in LUA
//
static const struct luaL_reg libsaliency_(Main__) [] =
{
  {"intImage",              libsaliency_(Main_intImage)},
  {"intHist",               libsaliency_(Main_intHist)},
  {"intHistLong",           libsaliency_(Main_intHistLong)},
  {"intHistPack",           libsaliency_(Main_intHistPack)},
  {"intAvg",                libsaliency_(Main_intAvg)},
  {"spatialMax",            libsaliency_(Main_spatialMax)},
  {"spatialOneOverMax",     libsaliency_(Main_spatialOneOverMax)},
  {"spatialMeanOverMax",    libsaliency_(Main_spatialMeanOverMax)},
  {"scaleSaliency",         libsaliency_(Main_scaleSaliency)},
  {"spatialEnt",            libsaliency_(Main_spatialEnt)},
  {"fastNMS",               libsaliency_(Main_fastNMS)},
  {"newNMS",                libsaliency_(Main_newNMS)},
  {NULL,NULL}
};
  
static void libsaliency_(Main_init) (lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, libsaliency_(Main__), "libsaliency");
  lua_pop(L,1);
}

#endif
