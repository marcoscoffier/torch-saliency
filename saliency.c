#include <TH.h>
#include <luaT.h>


#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libsaliency_(NAME) TH_CONCAT_3(libsaliency_, Real, NAME)


#include "generic/saliency.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libsaliency(lua_State *L)
{

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "libsaliency");

  libsaliency_FloatMain_init(L);
  libsaliency_DoubleMain_init(L);

  return 1;
}


