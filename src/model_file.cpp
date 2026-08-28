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
std::wstring to_wide(const char *p, size_t n) {
    const int w=MultiByteToWideChar(CP_UTF8,0,p,int(n),nullptr,0); std::wstring wide(size_t(w),L'\0');
    MultiByteToWideChar(CP_UTF8,0,p,int(n),wide.data(),w); return wide;
}
}

ModelFile::ModelFile(const wchar_t *index_path) {
    std::ifstream idx(index_path,std::ios::binary);
    if(!idx) throw std::runtime_error("cannot open Insignia index");
    idx.seekg(0,std::ios::end); const auto n=idx.tellg(); idx.seekg(0);
    std::vector<std::byte> blob(static_cast<size_t>(n)); idx.read(reinterpret_cast<char*>(blob.data()),n);
    const std::byte *p=blob.data(); const Header h=take<Header>(p);
    if(!std::memcmp(h.magic,"INSIDX01",8)&&h.version==1) {
        version_=1;
        payload_offset_=h.payload;
        const uint32_t path_n=take<uint32_t>(p); std::string path(reinterpret_cast<const char*>(p),path_n); p+=path_n;
        HANDLE f=CreateFileW(to_wide(path.data(),path_n).c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS,nullptr);
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
    } else if(!std::memcmp(h.magic,"INSIDX02",8)&&h.version==2) {
        // Layout mirrors tools/index27.py byte-for-byte: 12B magic+version, 9x u32 shape
        // header, u32 shard count, per shard {u32 path_len, u64 file_bytes, u32 crc,
        // path utf-8 relative to the index dir}, u32 tensor count, name-sorted
        // {u16 name_len, u8 dtype, u8 rank, name, u64 shape[rank], u16 shard,
        // u64 file_abs_off, u64 bytes}. (The 24B v1 Header over-read 12 bytes of the
        // shape header — rewind before parsing.)
        version_=2;
        p-=12;
        for(int i=0;i<9;i++) shape_hdr_[i]=take<uint32_t>(p);
        const wchar_t *slash=index_path; const wchar_t *end=index_path;
        while(*end) { if(*end==L'\\'||*end==L'/') slash=end+1; ++end; }
        index_dir_.assign(index_path,slash);
        const uint32_t shard_n=take<uint32_t>(p);
        shards_.resize(shard_n);
        for(uint32_t i=0;i<shard_n;i++) {
            const uint32_t plen=take<uint32_t>(p); const uint64_t fbytes=take<uint64_t>(p); const uint32_t crc=take<uint32_t>(p);
            std::string rel(reinterpret_cast<const char*>(p),plen); p+=plen;
            ShardInfo &s=shards_[i];
            s.path=index_dir_+to_wide(rel.data(),plen);
            s.bytes=fbytes; s.crc=crc;
        }
        const uint32_t tn=take<uint32_t>(p);   // h.count slot reused as tensor count
        tensors_.reserve(tn);
        for(uint32_t i=0;i<tn;i++) {
            const uint16_t name_n=take<uint16_t>(p); const DType dtype=take<DType>(p); const uint8_t rank=take<uint8_t>(p);
            TensorView t; t.name.assign(reinterpret_cast<const char*>(p),name_n); p+=name_n;
            t.dtype=dtype; t.shape.resize(rank); for(auto &d:t.shape)d=take<uint64_t>(p);
            t.shard=take<uint16_t>(p); t.off=take<uint64_t>(p); t.bytes=take<uint64_t>(p);
            if(t.shard>=shard_n||t.off+t.bytes>shards_[t.shard].bytes){close();throw std::runtime_error("tensor escapes shard: "+t.name);}
            tensors_.push_back(std::move(t));
        }
        shard_handles_.resize(shard_n,nullptr);
    } else throw std::runtime_error("bad Insignia index");
    std::sort(tensors_.begin(),tensors_.end(),[](const auto&a,const auto&b){return a.name<b.name;});
}
ModelFile::~ModelFile(){close();}
ModelFile::ModelFile(ModelFile&&o) noexcept { *this=std::move(o); }
ModelFile& ModelFile::operator=(ModelFile&&o) noexcept { if(this!=&o){close();file_=o.file_;mapping_=o.mapping_;base_=o.base_;mapped_bytes_=o.mapped_bytes_;payload_offset_=o.payload_offset_;version_=o.version_;std::memcpy(shape_hdr_,o.shape_hdr_,sizeof(shape_hdr_));shards_=std::move(o.shards_);shard_handles_=std::move(o.shard_handles_);index_dir_=std::move(o.index_dir_);tensors_=std::move(o.tensors_);o.file_=o.mapping_=nullptr;o.base_=nullptr;}return *this; }
void ModelFile::close() noexcept {
    for(void *h:shard_handles_) if(h) CloseHandle(h);
    shard_handles_.clear(); shards_.clear();
    if(base_)UnmapViewOfFile(base_);if(mapping_)CloseHandle(mapping_);if(file_)CloseHandle(file_);
    base_=nullptr;mapping_=file_=nullptr;
}
const TensorView *ModelFile::find(std::string_view name) const noexcept { auto it=std::lower_bound(tensors_.begin(),tensors_.end(),name,[](const TensorView&t,std::string_view n){return t.name<n;}); return it!=tensors_.end()&&it->name==name?&*it:nullptr; }

void *ModelFile::shard_handle(uint16_t shard) const {
    if(!shard_handles_[shard]) {
        HANDLE f=CreateFileW(shards_[shard].path.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(f==INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open shard "+std::to_string(shard));
        shard_handles_[shard]=f;
    }
    return shard_handles_[shard];
}
void ModelFile::read_into(const TensorView &t, void *dst) const { read_into_of(t, 0, t.bytes, dst); }
void ModelFile::read_into_of(const TensorView &t, uint64_t off, uint64_t len, void *dst) const {
    if (off + len > t.bytes) throw std::runtime_error("read_into_of out of tensor: " + t.name);
    if (version_ != 2) { std::memcpy(dst, t.data + off, size_t(len)); return; }
    HANDLE h = static_cast<HANDLE>(shard_handle(t.shard));
    OVERLAPPED ov{}; ov.Offset=uint32_t((t.off + off) & 0xffffffffull); ov.OffsetHigh=uint32_t((t.off + off) >> 32);
    uint8_t *out = static_cast<uint8_t *>(dst);
    uint64_t left = len;
    while (left) {
        DWORD want = left > 0x40000000 ? 0x40000000 : DWORD(left), got = 0;
        if (!ReadFile(h, out, want, &got, &ov) || !got) throw std::runtime_error("short read: " + t.name);
        out += got; left -= got; ov.Offset += got; if (ov.Offset < got) ++ov.OffsetHigh;   // manual carry: ov is not a pointer the driver updates
    }
}
}
