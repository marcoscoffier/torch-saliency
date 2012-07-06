require 'saliency'
require 'sys'
require 'image' 

op = xlua.OptionParser('%prog [options]')
op:option{'-f', '--full', action='store_true', dest='testFull', 
          help='test full matrix not just diagonal', default=false}
op:option{'-n', '--dontrun', action='store_true', dest='dontRun', 
          help='dont run the tests just load the functions', 
          default=false}
opt = op:parse()
op:summarize()

-- ++++++++++++++++++++++++++++++++++
-- 
-- Whole series of test functions for the different parts of the
-- integral hist images.
-- 
-- ++++++++++++++++++++++++++++++++++
function test_intImage(l)
   if not l then
      l = torch.randn(3,256,256)
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local ii = torch.Tensor(l:size())
   print("TESTING: libsaliency.intImage -- DIAGONAL ONLY")
   sys.tic()
   l.libsaliency.intImage(ii,l)
   print(
      string.format(" - time to compute intImage:           % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local s, v
   local errors = 0
   for i = 1,chan do
      for r = 1,dimr do 
         s = l:select(1,i):narrow(1,1,r):narrow(2,1,r):sum()
         v = ii[i][r][r]
         if (math.abs(s - v) > 1) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test (%d pts) intImage:   % 4.1f ms",
                    chan*dimr,sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,ii
end

function test_fullintImage(l)
   if not l then
      l = torch.randn(3,256,256)
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local ii = torch.Tensor(l:size())
   print("TESTING: libsaliency.intImage -- FULL")
   sys.tic()
   l.libsaliency.intImage(ii,l)
   print(
      string.format(" - time to make intImage:              % 4.1f ms",
                    sys.toc()*1000))
   local s, rs
   local errors = 0
   sys.tic()
   for i = 1,chan do
      for r = 1,dimr do 
         for c = 1,dimc do
            s = l:select(1,i):narrow(1,1,r):narrow(2,1,c):sum()
            rs = ii[i][r][c]
            if (math.abs(s - rs) > 1) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intImage:              % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors, ii
end


function test_intAvg(l,kr,kc,sr,sc)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not kr then
      kr = 5
   end
   if not kc then
      kc = 5
   end
   if not sr then
      sr = 1
   end
   if not sc then
      sc = 1
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local ii = torch.Tensor(l:size())
   local rr = torch.Tensor(l:size())
   print("TESTING: libsaliency.intAvg -- DIAGONAL ONLY")
   sys.tic() ;
   l.libsaliency.intImage(ii,l)
   print(
      string.format(" - time to compute intImage:           % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   ii.libsaliency.intAvg(rr,ii,kr,kc,sr,sc)
   print(
      string.format(" - time to compute intAvg:             % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local s,v
   local errors = 0
   for i = 1,chan do
      for r = 1,dimr-kr+1 do 
         if (r < dimc-kc+1) then
            s = l:select(1,i):narrow(1,r,kr):narrow(2,r,kc):mean()
            v = rr[i][r][r]
            if (math.abs(s - v) > 1e-8) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intAvg:                % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,ii,rr
end

function test_fullintAvg(l,kr,kc,sr,sc)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not kr then
      kr = 5
   end
   if not kc then
      kc = 5
   end
   if not sr then
      sr = 1
   end
   if not sc then
      sc = 1
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local ii = torch.Tensor(l:size())
   local rr = torch.Tensor(l:size())
   print("TESTING: libsaliency.intAvg -- FULL")
   sys.tic() ;
   l.libsaliency.intImage(ii,l)
   print(
      string.format(" - time to compute intImage:           % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   ii.libsaliency.intAvg(rr,ii,kr,kc,sr,sc)
   print(
      string.format(" - time to compute intAvg:             % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local s,v
   local errors = 0
   for i = 1,chan do
      for r = 1,dimr-kr+1 do 
         for c = 1,dimc-kc+1 do
            s = l:select(1,i):narrow(1,r,kr):narrow(2,c,kc):mean()
            v = rr[i][r][c]
            if (math.abs(s - v) > 1e-8) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intAvg:                % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,ii,rr
end


function test_intHist(l,nbins,lmin,lmax)
   if not l then 
      l = torch.randn(3,256,256)
   end
   if not lmin then
      lmin = l:min()
   end
   if not lmax then
      lmax = l:max()
   end
   if not nbins then
      nbins = 4
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   print("TESTING: libsaliency.intHist -- DIAGONAL ONLY") 
   sys.tic()
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   local s, rs
   sys.tic()
   local errors = 0;
   for i = 1,chan do
      for r = 1,dimr do 
         s = l:select(1,i):narrow(1,1,r):narrow(2,1,r):histc(nbins,lmin,lmax)
         rs = rr[i][r][r]
         local d = torch.abs(s - rs):sum()
         if ( d > 0) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test intHist:               % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr
end

function test_intLongHist(l,nbins,lmin,lmax)
   if not l then 
      l = torch.randn(3,256,256)
   end
   if not lmin then
      lmin = l:min()
   end
   if not lmax then
      lmax = l:max()
   end
   if not nbins then
      nbins = 4
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   rr = torch.Tensor(chan,dimr,dimc,nbins)
   rrl = torch.LongTensor(chan,dimr,dimc,nbins)
   ft = torch.Tensor(nbins)
   print("TESTING: libsaliency.intHist -- DIAGONAL ONLY") 
   sys.tic()
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   l.libsaliency.intHistLong(rrl,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHistLong:        % 4.1f ms",
                    sys.toc()*1000))
   local s, rs
   sys.tic()
   local errors = 0;
   for i = 1,chan do
      for r = 1,dimr do 
         rs = rr[i][r][r]
         ft:copy(rrl[i][r][r])
         local d = torch.abs(ft - rs):sum()
         if ( d > 0) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test intHist:               % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,rrl
end

function test_fullintHist(l,nbins)
   if not l then 
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 4
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local errors = 0
   print("TESTING: libsaliency.intHist -- FULL") 
   sys.tic()
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to make intHist:               % 4.1f ms",
                    sys.toc()*1000))
   local s, rs
   sys.tic()
   for i = 1,chan do
      for r = 1,dimr do 
         for c = 1,dimc do
            s = l:select(1,i):narrow(1,1,r):narrow(2,1,c):histc(nbins,lmin,lmax)
            rs = rr[i][r][c]
            local d = torch.abs(s - rs):sum()
            if ( d > 0) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intHist:               % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr
end

function test_intHistAvg(l,nbins,kr,kc,sr,sc)
   if not l then 
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 4
   end
   if not kr then
      kr = 5
   end
   if not kc then
      kc = 5
   end
   if not sr then
      sr = 1
   end
   if not sc then
      sc = 1
   end 
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local outr = (dimr - kr + 1)/sr
   local outc = (dimc - kc + 1)/sc
   ii = torch.Tensor(chan,dimr,dimc,nbins)
   aa = torch.Tensor(chan,outr,outc,nbins)
   print("TESTING: libsaliency.intAvg on intHist -- DIAGONAL ONLY") 
   sys.tic()
   l.libsaliency.intHist(ii,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   l.libsaliency.intAvg(aa,ii,kr,kc,sr,sc)
   print(
      string.format(" - time to compute avgHist:            % 4.1f ms",
                    sys.toc()*1000))
   local s, b, rs
   sys.tic()
   local errors = 0;
   for i = 1,chan do
      for r = 1,(dimr-kr)/sr + 1 do 
         if (r < (dimc-kc)/sc + 1) then
            b = l:select(1,i):narrow(1,r,kr):narrow(2,r,kc)
            s = b:histc(nbins,lmin,lmax):mul(1/(kr*kc))
            rs = aa[i][r][r]
            local d = torch.abs(s - rs):sum()
            if ( d > 0) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intAvg on intHist:     % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr
end

function test_fullintHistAvg(l,nbins,kr,kc,sr,sc)
   if not l then 
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 4
   end
   if not kr then
      kr = 5
   end
   if not kc then
      kc = 5
   end
   if not sr then
      sr = 1
   end
   if not sc then
      sc = 1
   end 
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local outr = (dimr - kr + 1)/sr
   local outc = (dimc - kc + 1)/sc
   local ii = torch.Tensor(chan,dimr,dimc,nbins)
   local aa = torch.Tensor(chan,outr,outc,nbins)
   print("TESTING: libsaliency.intAvg on intHist -- FULL") 
   sys.tic()
   l.libsaliency.intHist(ii,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   l.libsaliency.intAvg(aa,ii,kr,kc,sr,sc)
   print(
      string.format(" - time to compute avgHist:            % 4.1f ms",
                    sys.toc()*1000))
   local s, b, rs
   sys.tic()
   local errors = 0;
   for i = 1,chan do
      for r = 1,dimr-kr+1 do 
         for c = 1,dimc-kc+1 do
            b = l:select(1,i):narrow(1,r,kr):narrow(2,c,kc)
            s = b:histc(nbins,lmin,lmax):mul(1/(kr*kc))
            rs = aa[i][r][c]
            local d = torch.abs(s - rs):sum()
            if ( d > 0) then
               errors = errors + 1
            end
         end
      end
   end
   print(
      string.format(" - time to test intAvg on intHist:     % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr
end

function test_spatialMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(chan,dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialMax -- DIAGONAL ONLY")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialMax(mm,rr)
   print(
      string.format(" - time to compute spatialMax:         % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v,tmp
   local errors = 0
   for r = 1,dimr do
      m = rr[1][r][r]:max()
      -- max is over all channels
      for i = 2,chan do 
         tmp = rr[i][r][r]:max()
         if (tmp > m) then m = tmp end
      end
      v = mm[r][r]
      if (math.abs(m - v) > 1e-8) then
         errors = errors + 1
      end 
   end
   print(
      string.format(" - time to test spatialMax:            % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_fullspatialMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(chan,dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialMax -- FULL")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialMax(mm,rr)
   print(
      string.format(" - time to compute spatialMax:         % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v,tmp
   local errors = 0
   for r = 1,dimr do 
      for c = 1,dimc do
         m = rr[1][r][c]:max()
         -- max is over all channels
         for i = 2,chan do 
            tmp = rr[i][r][c]:max()
            if (tmp > m) then m = tmp end
         end
         v = mm[r][c]
         if (math.abs(m - v) > 1e-8) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test spatialMax:            % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_spatialOneOverMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(chan,dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialOneOverMax -- DIAGONAL ONLY")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialOneOverMax(mm,rr)
   print(
      string.format(" - time to compute spatialOneOverMax:  % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v
   local errors = 0
   for r = 1,dimr do
      m = 1/rr[1][r][r]:max()
      -- 1/max is summed over all channels
      for i = 2,chan do 
         m = m + 1/rr[i][r][r]:max()         
      end
      v = mm[r][r]
      if (math.abs(m - v) > 1e-8) then
         errors = errors + 1
      end 
   end
   print(
      string.format(" - time to test spatialOneOverMax:     % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_fullspatialOneOverMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialOneOverMax -- FULL")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialOneOverMax(mm,rr)
   print(
      string.format(" - time to compute spatialOneOverMax:  % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v,tmp
   local errors = 0
   for r = 1,dimr do 
      for c = 1,dimc do
         m = 1/rr[1][r][c]:max()
         -- 1/max is summed over all channels
         for i = 2,chan do 
            m = m + 1/rr[i][r][c]:max()         
         end
         v = mm[r][c]
         if (math.abs(m - v) > 1e-8) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test spatialOneOverMax:     % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_spatialMeanOverMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialMeanOverMax -- DIAGONAL ONLY")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialMeanOverMax(mm,rr)
   print(
      string.format(" - time to compute spatialMeanOverMax: % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v
   local errors = 0
   for r = 1,dimr do
      m = rr[1][r][r]:mean()/rr[1][r][r]:max() 
      -- mean/max is summed over all channels
      for i = 2,chan do 
         m = m + rr[i][r][r]:mean()/rr[i][r][r]:max()
      end
      v = mm[r][r]
      if (math.abs(m - v) > 1e-8) then
         errors = errors + 1
      end 
   end
   print(
      string.format(" - time to test spatialMeanOverMax:    % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_fullspatialMeanOverMax(l,nbins)
   if not l then
      l = torch.randn(3,256,256)
   end
   if not nbins then
      nbins = 8
   end
   local chan = l:size(1)
   local dimr = l:size(2)
   local dimc = l:size(3)
   local lmin = l:min()
   local lmax = l:max()
   local rr = torch.Tensor(chan,dimr,dimc,nbins)
   local mm = torch.Tensor(chan,dimr,dimc)
   local errors = 0
   print("TESTING: libsaliency.spatialMeanOverMax -- FULL")
   sys.tic() ;
   l.libsaliency.intHist(rr,l,nbins,lmin,lmax)
   print(
      string.format(" - time to compute intHist:            % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   rr.libsaliency.spatialMeanOverMax(mm,rr)
   print(
      string.format(" - time to compute spatialMeanOverMax: % 4.1f ms",
                    sys.toc()*1000))
   sys.tic()
   local m,v,tmp
   local errors = 0
   for r = 1,dimr do 
      for c = 1,dimc do
         m = rr[1][r][c]:mean()/rr[1][r][c]:max() 
         -- mean/max is summed over all channels
         for i = 2,chan do 
            m = m + rr[i][r][c]:mean()/rr[i][r][c]:max()
         end
         v = mm[r][c]
         if (math.abs(m - v) > 1e-8) then
            errors = errors + 1
         end
      end
   end
   print(
      string.format(" - time to test spatialMeanOverMax:    % 4.1f ms",
                    sys.toc()*1000))
   print(" - found "..errors.." errors")
   return errors,rr,mm
end

function test_getMaxN_simpleGaussian(gsize,msize)
   if not gsize then
      gsize = 11
   end
   if not msize then
      msize = 5
   end
   local img = image.gaussian(gsize)
   local m,p = saliency.getMaxNew(img,msize,gsize*gsize)
   local rgbimg = torch.Tensor(3,gsize,gsize)
   
   rgbimg:select(1,1):copy(img)
   rgbimg:select(1,2):copy(img)
   rgbimg:select(1,3):copy(img)
   
   for i = 1,p do
      print("Found: ".. m[i][1]..", "..m[i][2])
      rgbimg[1][m[i][1]][m[i][2]] = 0
      rgbimg[2][m[i][1]][m[i][2]] = 1
      rgbimg[3][m[i][1]][m[i][2]] = 0
   end

   return rgbimg,m,p
end   


function test_getMaxN_multiGaussian(ng,gsize,imsize,msize)
   if not ng then
      ng = 25
   end
   if not gsize then
      gsize = 17
   end
   if not imsize then
      imsize = 256
   end
   if not msize then
      msize = 5
   end

   print("TESTING: saliency.getMaxN")
   local img = torch.Tensor(imsize,imsize):zero()
   local gsizes = torch.rand(ng,1):mul((gsize-5)*0.5):floor():mul(2):add(5)
   local gweight = torch.rand(ng,1):add(0.1)
   local xy = torch.rand(ng,2):mul(imsize-gsize-1):add(1):floor()
   for i = 1,ng do
      img:narrow(1,xy[i][1],gsizes[i][1]):narrow(2,xy[i][2],gsizes[i][1]):add(image.gaussian(gsizes[i][1]))
   end
   sys.tic()
   -- m,p = saliency.getMaxNew(img,msize,ng)
   local m,mi,p = saliency.getMax(img,msize,ng)
   print(
      string.format(" - time to compute getMaxN:            % 4.1f ms",
                    sys.toc()*1000))

   local rgbimg = torch.Tensor(3,imsize,imsize)
   rgbimg:select(1,1):copy(img)
   rgbimg:select(1,2):copy(img)
   rgbimg:select(1,3):copy(img)
   
   local w      = 5
   local hw     = math.floor(w*0.5)

   sys.tic()
   for i = 1,p do
      local xr= rgbimg:narrow(2,m[i][2]-hw,w):narrow(3,m[i][1],1)
      xr:zero()
      xr:select(1,2):add(1)
      local xc = rgbimg:narrow(2,m[i][2],1):narrow(3,m[i][1]-hw,w)
      xc:zero()
      xc:select(1,2):add(1)
   end
   print(
      string.format(" - time to test getMaxN:               % 4.1f ms",
                    sys.toc()*1000))
   image.display{image={rgbimg},zoom=2}
   return img,m,p,xy
end   


-- these are all the accepted tests
if not opt.dontRun then 
   if (test_intImage() > 0) 
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then
      if (test_fullintImage() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_intAvg() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then
      if (test_fullintAvg() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_intHist() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then 
      if (test_fullintHist() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_spatialMax() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then
      if (test_fullspatialMax() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_spatialOneOverMax() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then
      if (test_fullspatialOneOverMax() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_intHistAvg() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then 
      if (test_fullintHistAvg() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   if (test_spatialMeanOverMax() > 0)
   then print("**ERROR**") 
   else print("**OK**")end

   if opt.testFull then
      if (test_fullspatialMeanOverMax() > 0)
      then print("**ERROR**") 
      else print("**OK**")end
   end

   img,m,p,xy = test_getMaxN_multiGaussian()

end


