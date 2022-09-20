/*!
 * \file rt_module.cc
 */



#include "rt_module.h" 
 

namespace tvm {
namespace runtime {

YoModule::YoModule(const std::string& name, const std::string& dev_type, const std::string& code,
                   const std::string& dev_cfg)
    : name_(name),
      dev_type_(dev_type), 
      code_(code),
      dev_cfg_(dev_cfg), 
      device_("yoDev") {}

Module YoModuleLoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string name;
  std::string dev_type; 
  std::string code;
  std::string dev_cfg;
  stream->Read(&name);
  stream->Read(&dev_type);  
  stream->Read(&code);
  stream->Read(&dev_cfg);
  auto exec =
      make_object<YoModule>(name, dev_type, code, dev_cfg);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_YoModule").set_body_typed(YoModuleLoadFromBinary);

void YoModule::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(name_);
  stream->Write(dev_type_);  
  stream->Write(code_);
  stream->Write(dev_cfg_);
}

PackedFunc YoModule::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  std::cout<< "====== In YoModule GetFunction: " << name << std::endl;

  if (name == "get_symbol") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->name_; });
  } else if (name == "get_const_vars") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
          Array<String> const_vars;
          *rv = const_vars; });

  } else if ("__init_" + this->name_ == name) {
    // The function to initialize constant tensors.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1U); 
      *rv = 0;
    });
  } else if (this->name_ == name) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

        std::cout << "====== Run instructions:" << code_ << std::endl; 
    });
  } else {
    return PackedFunc();
  }
}

Module YoModuleCreate(const std::string& name, const std::string& dev_type,
                      const std::string& code, const std::string& dev_cfg) { 
  auto exec = make_object<YoModule>(name, dev_type, code, dev_cfg);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.yo_module.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = YoModuleCreate(args[0], args[1], args[2], args[3]);
});
}  // namespace runtime
}  // namespace tvm
