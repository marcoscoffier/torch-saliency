package = "saliency"
version = "1.0-0"

source = {
   url = "git://github.com/marcoscoffier/torch-saliency",
   tag = "1.0-0"
}

description = {
   summary = "Saliency operator and integral images",
   detailed = [[
   ]],
   homepage = "https://github.com/marcoscoffier/torch-saliency",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
