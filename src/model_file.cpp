#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "insignia_model.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace insignia {
namespace {
template<class T> T take(const std::byte *&p) { T x; std::memcpy(&x,p,sizeof(x)); p+=sizeof(x); return x; }
#pragma pack(push,1)
struct Header { char magic[8]; uint32_t version; uint32_t count; uint64_t payload; };
#pragma pack(pop)
}

ModelFile::ModelFile(const wchar_t *index_path) {
    std::ifstream idx(index_path,std::ios::binary);
    if(!idx) throw std::runtime_error("cannot open Insignia index");
    idx.seekg(0,std::ios::end); const auto n=idx.tellg(); idx.seekg(0);
    std::vector<std::byte> blob(static_cast<size_t>(n)); idx.read(reinterpret_cast<char*>(blob.data()),n);
    const std::byte *p=blob.data(); const Header h=take<Header>(p);
    if(std::memcmp(h.magic,"INSIDX01",8)||h.version!=1) throw std::runtime_error("bad Insignia index");
    payload_offset_=h.payload;
    const uint32_t path_n=take<uint32_t>(p); std::string path(reinterpret_cast<const char*>(p),path_n); p+=path_n;
    const int wide_n=MultiByteToWideChar(CP_UTF8,0,path.data(),path_n,nullptr,0); std::wstring wide(wide_n,L'\0'); MultiByteToWideChar(CP_UTF8,0,path.data(),path_n,wide.data(),wide_n);
    HANDLE f=CreateFileW(wide.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS,nullptr);
    if(f==INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open model payload"); file_=f;
    LARGE_INTEGER size{}; if(!GetFileSizeEx(f,&size)){close();throw std::runtime_error("cannot size model payload");} mapped_bytes_=size.QuadPart;
    HANDLE m=CreateFileMappingW(f,nullptr,PAGE_READONLY,0,0,nullptr); if(!m){close();throw std::runtime_error("cannot map model payload");} mapping_=m;
    base_=static_cast<const std::byte*>(MapViewOfFile(m,FILE_MAP_READ,0,0,0)); if(!base_){close();throw std::runtime_error("cannot view model payload");}
    tensors_.reserve(h.count);
    for(uint32_t i=0;i<h.count;i++) {
        const uint16_t name_n=take<uint16_t>(p); const DType dtype=take<DType>(p); const uint8_t rank=take<uint8_t>(p); const uint64_t off=take<uint64_t>(p); const uint64_t bytes=take<uint64_t>(p);
        TensorView t; t.name.assign(reinterpret_cast<const char*>(p),name_n); p+=name_n; t.dtype=dtype; t.shape.resize(rank); for(auto &d:t.shape)d=take<uint64_t>(p); t.data=base_+payload_offset_+off; t.bytes=bytes;
        if(payload_offset_+off+bytes>mapped_bytes_){close();throw std::runtime_error("tensor escapes model mapping");} tensors_.push_back(std::move(t));
    }
    std::sort(tensors_.begin(),tensors_.end(),[](const auto&a,const auto&b){return a.name<b.name;});
}
ModelFile::~ModelFile(){close();}
ModelFile::ModelFile(ModelFile&&o) noexcept { *this=std::move(o); }
ModelFile& ModelFile::operator=(ModelFile&&o) noexcept { if(this!=&o){close();file_=o.file_;mapping_=o.mapping_;base_=o.base_;mapped_bytes_=o.mapped_bytes_;payload_offset_=o.payload_offset_;tensors_=std::move(o.tensors_);o.file_=o.mapping_=nullptr;o.base_=nullptr;}return *this; }
void ModelFile::close() noexcept { if(base_)UnmapViewOfFile(base_);if(mapping_)CloseHandle(mapping_);if(file_)CloseHandle(file_);base_=nullptr;mapping_=file_=nullptr; }
const TensorView *ModelFile::find(std::string_view name) const noexcept { auto it=std::lower_bound(tensors_.begin(),tensors_.end(),name,[](const TensorView&t,std::string_view n){return t.name<n;}); return it!=tensors_.end()&&it->name==name?&*it:nullptr; }
}
