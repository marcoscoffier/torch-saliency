require 'torch'
require 'dok'
require 'inline'

saliency = {}

-- load C lib
require 'libsaliency'


function saliency.high_entropy_features (...)
   local _, img, kr, kc, sr, sc, nscale, scalefactor, nbins, entropyType, nmswinsize, npts = dok.unpack(
      {...},
      'saliency.high_entropy_features',
      '[[ Implements a saliency detector based on entropy of histogram of input.  And searches for a saliency across scales. ]]',
      {arg='img', type='torch.Tensor',
       help='torch.Tensor image (though can have multiple channels', 
       req=true},
      {arg='kr', type='number',
       help='size of hist window in pixels (row)', default=5},
      {arg='kc', type='number',
       help='size of hist window in pixels (col)', default=5},
      {arg='sr', type='number',
       help='step of hist window in pixels (row)', default=1},
      {arg='sc', type='number',
       help='step of hist window in pixels (col)', default=1}, 
      {arg='nscale', type='number',
       help='how many scales?', default=3},
      {arg='scalefactor', type='number',
       help='what is scaling factor between scales', default=1.2},
      {arg='nbins', type='number',
       help='how many bins in the histograms ?', default=8},
      {arg='entropyType', type='string',
       help='compute true entropy or fast approx (entropy or meanOverMax)',
       default='meanovermax'},
      {arg='nmswinsize', type='number',
       help='what window over which to do non maximal suppression', 
       default=5},
      {arg='npts', type='number',
       help='how many points do we return ?', default=30}
   )

   local t = torch.Timer()
   local w  = img:size(3)
   local h  = img:size(2)
   local ii = torch.Tensor(img:size(1),h,w,nbins)
   local rr = torch.Tensor(img:size(1),h-kr+1,w-kc+1,nbins)
   local sm = torch.Tensor(nscale,h,w):zero()
   local m  = torch.Tensor(h,w):zero()
   local mpoints_out  = torch.Tensor(npts,2)

   -- compute once for all images
   local tTotal = t:time().real
   img.libsaliency.intHist(ii,img,nbins)
   local tintHist = t:time().real - tTotal
   local tAvg = {}
   local tEnt = {}
   -- FIXME optimize this make a multi-scale in C.
   for i = 1,nscale do
      tAvg[i] = t:time().real 
      local lkr = math.floor(kr * scalefactor^(i-1))
      local lkc = math.floor(kc * scalefactor^(i-1))
      print(' - computing ('..lkr..','..lkc..')')
      img.libsaliency.intAvg(rr,ii,lkr,lkc,sr,sc)
      tAvg[i] = t:time().real - tAvg[i]
      tEnt[i] = t:time().real
      local nch = rr:size(1)
      local nr  = rr:size(2)
      local nc  = rr:size(3)
      local outsizer = math.floor(0.5+(h-lkr+1)/sr)
      local outsizec = math.floor(0.5+(w-lkc+1)/sc)
      local fm  = torch.Tensor(outsizer,outsizec)
      if entropyType == 'entropy' then 
         img.libsaliency.spatialEnt(fm,rr);
      elseif entropyType == 'MeanOver' then
         img.libsaliency.spatialMeanOverMax(fm,rr);
      else 
         img.libsaliency.spatialOneOverMax(fm,rr);
      end
      -- print("["..i.."] min: "..fm:min().." max: "..fm:max())
      sm:select(1,i):narrow(1,math.floor(lkr/2),outsizer):narrow(2,math.floor(lkc/2),outsizec):copy(fm)
      tEnt[i] = t:time().real - tEnt[i] 
   end
   local tScaleSaliency = t:time().real
   -- compute saliency across scales
   if nscale == 1 then
      m = sm:select(1,1)
   else
      m.libsaliency.scaleSaliency(m,sm)
      -- y = H*S
      m:cmul(sm:sum(1):select(1,1))
   end
   tScaleSaliency = t:time().real - tScaleSaliency
   local tclearB = t:time().real
   local mkr = math.floor(kr*scalefactor^(nscale-1))
   local mkc = math.floor(kc*scalefactor^(nscale-1))
   local okr = math.floor(mkr/2)
   local okc = math.floor(mkc/2)
   local opr = math.floor(0.5+(h-mkr+1)/sr)
   local opc = math.floor(0.5+(w-mkc+1)/sc)
   local tmpm  = torch.Tensor(m:size()):zero()
   tmpm:narrow(1,okr,opr):narrow(2,okc,opc):fill(1)
   -- clear the borders
   m:cmul(tmpm)
   tclearB = t:time().real - tclearB
   tTotal = t:time().real - tTotal
   print(string.format(" - intHist %2d bins: %2.3fs (%2.1f%%)",
                       nbins,tintHist,tintHist/tTotal*100))
   print(" - scales:")
   for i = 1,nscale do
      print(string.format(" -- SpatialAvg[%d]:  %2.3fs (%2.1f%%)",
            i,tAvg[i],tAvg[i]/tTotal*100))
      print(string.format(" -- SpatialEnt[%d]:  %2.3fs (%2.1f%%)",
            i,tEnt[i],tEnt[i]/tTotal*100))
   end
   print(string.format(" - scale saliency:  %2.3fs (%2.1f%%)",
                       tScaleSaliency,tScaleSaliency/tTotal*100))
   print(string.format(" - clear border:  %2.3fs (%2.1f%%)",
                       tclearB,tclearB/tTotal*100))
   print(string.format("       Total time:  %2.3fs (%2.1f%%)",
                       tTotal,tTotal/tTotal*100))
   return m,sm
end

function saliency.spent_pxdiff_motion (...)
   local _, img, imgprev, mix, kr, kc, sr, sc, nscale, scalefactor, nbins, entropyType, nmswinsize, npts, histsize = dok.unpack(
      {...},
      'saliency.spent_pxdiff_motion',
      '[[ Spatial Entropy + Pixel difference ]]',
      {arg='img', type='torch.Tensor',
       help='torch.Tensor image (though can have multiple channels', 
       req=true},
      {arg='imgprev', type='table',
       help='table of previous spatial saliencies'},
      {arg='mix', type='number',
       help='percent to blend motion and spatial saliency', default=0.5},
      {arg='kr', type='number',
       help='size of hist window in pixels (row)', default=5},
      {arg='kc', type='number',
       help='size of hist window in pixels (col)', default=5},
      {arg='sr', type='number',
       help='step of hist window in pixels (row)', default=1},
      {arg='sc', type='number',
       help='step of hist window in pixels (col)', default=1}, 
      {arg='nscale', type='number',
       help='how many scales?', default=3},
      {arg='scalefactor', type='number',
       help='what is scaling factor between scales', default=1.2},
      {arg='nbins', type='number',
       help='how many bins in the histograms ?', default=8},
      {arg='entropyType', type='string',
       help='compute true entropy or fast approx (entropy or meanOverMax)',
       default='meanovermax'},
      {arg='nmswinsize', type='number',
       help='what window over which to do non maximal suppression', 
       default=5},
      {arg='npts', type='number',
       help='how many points do we return ?', default=30},
      {arg='histsize', type='number',
       help='how many images do we compare', default=2}
   )
   -- compute spatial saliency on new image
   m,sm = saliency.high_entropy_features(
      img, kr, kc, sr, sc, 
      nscale, scalefactor, nbins, entropyType, nmswinsize, npts)
   -- ring buffer
   if ((not imgprev) or (imgprev == {})) then
      return {{m,sm}}
   else 
      table.insert(imgprev,1,{m,sm,mxy,mv,np})
      if (#imgprev > histsize) then
         for i = histsize+1,#imgprev do
            table.remove(imgprev,i)
         end
      end
      return imgprev
   end
end

function saliency.high_entropy_motion_features (...)
   local _, img, imgprev, kr, kc, sr, sc, nscale, scalefactor, nbins, entropyType, nmswinsize, npts, histsize = dok.unpack(
      {...},
      'saliency.high_entropy_motion_features',
      '[[ Implements a saliency detector based on entropy of histogram of input.  And searches for a saliency across scales and time. ]]',
      {arg='img', type='torch.Tensor',
       help='torch.Tensor image (though can have multiple channels', 
       req=true},
      {arg='imgprev', type='table',
       help='table of previous spatial saliencies'},
      {arg='kr', type='number',
       help='size of hist window in pixels (row)', default=5},
      {arg='kc', type='number',
       help='size of hist window in pixels (col)', default=5},
      {arg='sr', type='number',
       help='step of hist window in pixels (row)', default=1},
      {arg='sc', type='number',
       help='step of hist window in pixels (col)', default=1}, 
      {arg='nscale', type='number',
       help='how many scales?', default=3},
      {arg='scalefactor', type='number',
       help='what is scaling factor between scales', default=1.2},
      {arg='nbins', type='number',
       help='how many bins in the histograms ?', default=8},
      {arg='entropyType', type='string',
       help='compute true entropy or fast approx (entropy or meanOverMax)',
       default='meanovermax'},
      {arg='nmswinsize', type='number',
       help='what window over which to do non maximal suppression', 
       default=5},
      {arg='npts', type='number',
       help='how many points do we return ?', default=30},
      {arg='histsize', type='number',
       help='how many images do we compare', default=2}
   )
   -- compute spatial saliency on new image
   m,sm = saliency.high_entropy_features(
      img, kr, kc, sr, sc, 
      nscale, scalefactor, nbins, entropyType, nmswinsize, npts)
   -- ring buffer
   if ((not imgprev) or (imgprev == {})) then
      return {{m,sm}}
   else 
      table.insert(imgprev,1,{m,sm,mxy,mv,np})
      if (#imgprev > histsize) then
         for i = histsize+1,#imgprev do
            table.remove(imgprev,i)
         end
      end
      return imgprev
   end
end

-- Non-maximal suppression: takes the max of each row and the max of
-- each col and places in a buffer if this value = the actual value
-- then you have a local max at this position.

function saliency.getMax(m,windowsize,npts)
   if not windowsize then
      windowsize = 5
   end
   if not npts then
      npts = 10
   end
   local t      = torch.Timer()
   bufxy,bufval,k = 
      m.libsaliency.nonMaximalSuppression(m,windowsize,npts)
   return bufxy,bufval,k
end

-- sm is a 3D tensor nscales x width x height. This function sums the
-- abs difference between the scales and multiplies this by the sum of
-- the saliencies at each scale.
function saliency.scaleSaliency(sm)
   local m = torch.Tensor(sm:size(2),sm:size(3))
   -- compute saliency across scales
   if sm:size(1) == 1 then
      m = sm:select(1,1)
   else
      m.libsaliency.scaleSaliency(m,sm)
      -- y = H*S
      m:cmul(sm:sum(1):select(1,1))
   end
   return m
end

-- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Lua versions of functions which have been recoded in c
-- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

-- slow lua version replaced by the c version in saliency.c
function integer_img(img)
   local r = torch.Tensor():resizeAs(img)
   for i = 1,img:size(1) do
      -- special copy for first elem first row first col
      r[i][1][1] = img[i][1][1]
      -- special loop for first col
      for k = 2,img:size(3) do
         r[i][1][k] = img[i][1][k] + r[i][1][k-1]
      end
      for j = 2,img:size(2) do
         -- special copy for first row
         r[i][j][1] = img[i][j][1] + r[i][j-1][1]
         row = img[i][j][1]
         for k = 2,img:size(3) do
            row = row + img[i][j][k]
            r[i][j][k] = row + r[i][j-1][k] 
         end
      end
   end
   return r
end

-- lua version of the above code used for testing.
function saliency.getMaxLua(m,windowsize,npts)
   if not windowsize then
      windowsize = 5
   end
   if not npts then
      npts = 10
   end
   if (npts < 1) then 
      npts = m:size(1)*m:size(2)
   end
   local t      = torch.Timer()
   local tTotal = t:time().real
   local outr   = m:size(1) - windowsize + 1
   local outc   = m:size(2) - windowsize + 1
   local outxy  = torch.Tensor(npts,2)
   local hw     = math.floor(windowsize*0.5)
   local bufidx = torch.Tensor(outr*outc,2)
   local bufval = torch.Tensor(outr*outc)
   local tMaxR  = t:time().real
   local mr     = m:unfold(1,windowsize,1):max(3)
   local tMaxR  = t:time().real - tMaxR
   local tMaxC  = t:time().real
   local mc     = m:unfold(2,windowsize,1):max(3)
   local tMaxC  = t:time().real - tMaxC
   local k      = 0
   local tMaxEq = t:time().real
   -- get all maxima
   for i = 1,outr do
      for j = 1,outc do
         local offr = i+hw
         local offc = j+hw
         if (((mr[i][offc][1] == mc[offr][j][1]) and 
           (mr[i][offc][1] > 0)) and
          (mr[i][offc][1] == m[offr][offc])) then
            k = k + 1
            -- opencv ordering
            bufidx[k][1] = offc
            bufidx[k][2] = offr
            bufval[k]    = m[offr][offc]
         end
      end
   end
   local tMaxEq  = t:time().real - tMaxEq
   print(" - Found "..k.." points of "..npts)
   local tSort = t:time().real
   local outval = {}
   local outidx = {}
   if (k>0) then
      bufidx = bufidx:narrow(1,1,k)
      bufval = bufval:narrow(1,1,k)
      outval, outidx   = torch.sort(bufval,1,true)
      points = math.min(k,npts)
      -- copy x,y of top points
      for i = 1,points do
         outxy:select(1,i):copy(bufidx:select(1,outidx[i]))
      end
      -- make sure to narrow out if points is shorter than originally
      -- requested.
      if (not (points == npts)) then
         outxy = outxy:narrow(1,1,points)
      end
   else 
      points = 0
      outxy = torch.Tensor()
   end
   tSort  = t:time().real - tSort
   tTotal = t:time().real - tTotal
   print(string.format(" - MaxR:  %2.3fs (%2.1f%%)",
                       tMaxR,tMaxR/tTotal*100))
   print(string.format(" - MaxC:  %2.3fs (%2.1f%%)",
                       tMaxC,tMaxC/tTotal*100))
   print(string.format(" - MaxEq:  %2.3fs (%2.1f%%)",
                       tMaxEq,tMaxEq/tTotal*100))
   print(string.format(" - just sort:       %2.3fs (%2.1f%%)",
                       tSort, tSort/tTotal*100))
   print(string.format("       Total time:  %2.3fs (%2.1f%%)",
                       tTotal,tTotal/tTotal*100))

   return outxy, outval, points
end

