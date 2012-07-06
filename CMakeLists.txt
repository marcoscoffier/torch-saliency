CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(luasrc init.lua)
SET(src saliency.c)

FIND_PACKAGE(OpenMP)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

FIND_LIBRARY(THOMP THOmp)

ADD_TORCH_PACKAGE(saliency "${src}" "${luasrc}" "Saliency using Entropy on Integer (integral) Histograms")
TARGET_LINK_LIBRARIES(saliency luaT TH ${THOMP})
