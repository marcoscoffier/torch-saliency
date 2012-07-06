#include <TH.h>
#include <luaT.h>


#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define libsaliency_(NAME) TH_CONCAT_3(libsaliency_, Real, NAME)

static const void* torch_LongTensor_id   = NULL;
static const void* torch_FloatTensor_id  = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/saliency.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libsaliency(lua_State *L)
{

  torch_LongTensor_id   = luaT_checktypename2id(L, "torch.LongTensor");
  torch_FloatTensor_id  = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  libsaliency_FloatMain_init(L);
  libsaliency_DoubleMain_init(L);

  luaL_register(L, "libsaliency.double", libsaliency_DoubleMain__);
  luaL_register(L, "libsaliency.float", libsaliency_FloatMain__);

  return 1;
}


