#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using google::protobuf::TextFormat;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

class ProtobufUtil {
 public:

  static bool ParseFromASCIIFile(const std::string &file_name,
                                 google::protobuf::Message *message) {
    int file_descriptor = open(file_name.c_str(), O_RDONLY);
    if (file_descriptor < 0) {
      std::cerr << "Failed to open file " << file_name << " in text mode." << std::endl;
      return false;
    }

    ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
    bool success = TextFormat::Parse(input, message);
    if (!success) {
      std::cerr << "Failed to parse file " << file_name << " as text proto." << std::endl;
    }
    delete input;
    close(file_descriptor);
    return success;
  }

  static bool ParseFromBinaryFile(const std::string &file_name,
                                  google::protobuf::Message *message) {
    std::fstream input(file_name, std::ios::in | std::ios::binary);
    if (!input.good()) {
      std::cerr << "Failed to open file " << file_name << " in binary mode." << std::endl;
      return false;
    }
    if (!message->ParseFromIstream(&input)) {
      std::cerr << "Failed to parse file " << file_name << " as binary proto." << std::endl;
      return false;
    }
    return true;
  }

  static bool SaveToASCIIFile(const google::protobuf::Message &message,
                              const std::string &file_name) {
    int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
    if (fd < 0) {
      std::cerr << "Unable to open file " << file_name << " to write." << std::endl;
      return false;
    }
    ZeroCopyOutputStream *output = new FileOutputStream(fd);
    bool success = TextFormat::Print(message, output);
    delete output;
    close(fd);
    return success;
  }

  static bool SaveToBinaryFile(const google::protobuf::Message &message,
                               const std::string &file_name) {
    std::fstream output(file_name,
                        std::ios::out | std::ios::trunc | std::ios::binary);
    return message.SerializeToOstream(&output);
  }
};
