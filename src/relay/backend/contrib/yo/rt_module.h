#include <tvm/runtime/registry.h>

#include <fstream>
#include <streambuf>
#include <string>
#include <vector>


namespace tvm {
namespace runtime {

class YoModule : public ModuleNode {
 public:
  /*!
   * \brief Create My dev module
   * \param name The name of the function.
   * \param dev_type The device type. 
   * \param code The serialized device code.
   * \param dev_cfg The JSON serialized device config. 
   */
  YoModule(const std::string& name, const std::string& dev_type, const std::string& code,
                  const std::string& dev_cfg);

  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const { return "Yo"; }

  /*!
   * \brief Serialize the content of the pyxir directory and save it to
   *        binary stream.
   * \param stream The binary stream to save to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

 private: 
  std::string name_; 
  std::string dev_type_; 
  std::string code_; 
  std::string dev_cfg_; 
  std::string device_;
  
};

}
}