
/*!
 * \file src/relay/backend/contrib/yo/cfg_yo.cc
 * \brief Register Yo codegen options. Main codegen is implemented in python.
 */

#include <tvm/ir/transform.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace yo {

/*! \brief Attributes to store the compiler options for Yo */
struct YoCompilerCfgNode : public tvm::AttrsNode<YoCompilerCfgNode> {
  String target;
  TVM_DECLARE_ATTRS(YoCompilerCfgNode, "ext.attrs.YoCompilerCfgNode") {
    //   std::cout << "hey" << std::endl;
    TVM_ATTR_FIELD(target).describe("yo target").set_default("");
  }
};

class YoCompilerCfg : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(YoCompilerCfg, Attrs,
                                            YoCompilerCfgNode);
};

TVM_REGISTER_NODE_TYPE(YoCompilerCfgNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.yo.options", YoCompilerCfg);

/*! \brief The target yo device */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.yo.options.target", String); 


}  // namespace yo
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
