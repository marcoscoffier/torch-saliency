torch-saliency
==============

code and tools around integral images.

A library for finding interest points based on fast integral
historgrams.  The saliency is computed based on Kadir 2004.  The
largest change in entropy between scales is the most salient.

several functions can be of more use generally.  The interface will be
cleaned up shortly. Need to add a lua interface and help to the
working C functions.

compute an integer image ii from l <KxHxW>
l.libsaliency.intImage(ii,l)

compute an integer histogram rr from l <KxHxW>
l.libsaliency.intHist(rr,l,nbins,lmin,lmax)

average windows < kr x kc > steped < sr x sc > across the integral image <ii>
l.libsaliency.intAvg(aa,ii,kr,kc,sr,sc)

Max, OneOverMax, MeanOverMax

rr.libsaliency.spatialMax(mm,rr)
rr.libsaliency.spatialOneOverMax(mm,rr)
rr.libsaliency.spatialMeanOverMax(mm,rr)

Non Maximal suppression code: 

local m,mi,p = saliency.getMax(img,msize,ng)

There is an extensive regression test: 

test.lua
