#include "insignia_storage.hpp"
#include <algorithm>
#include <stdexcept>
namespace insignia {
static void check(cudaError_t e,const char*what){if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));}
TieredStorage::TieredStorage(const ModelFile&m,uint64_t b,cudaStream_t s):model_(m),budget_(b),stream_(s){}
TieredStorage::~TieredStorage(){try{clear();}catch(...){}}
void TieredStorage::make_room(uint64_t bytes){if(bytes>budget_)throw std::runtime_error("tensor exceeds device budget");while(used_+bytes>budget_){auto victim=entries_.end();for(auto it=entries_.begin();it!=entries_.end();++it)if(it->second.device&&it->second.pins==0&&(victim==entries_.end()||it->second.tick<victim->second.tick))victim=it;if(victim==entries_.end())throw std::runtime_error("device budget exhausted by pinned tensors");check(cudaStreamSynchronize(stream_),"synchronize before eviction");check(cudaFree(victim->second.device),"cudaFree");used_-=victim->second.bytes;entries_.erase(victim);}}
DeviceView TieredStorage::acquire(std::string_view name){auto key=std::string(name);auto it=entries_.find(key);if(it!=entries_.end()){it->second.pins++;it->second.tick=++tick_;return {it->second.device,it->second.bytes,it->second.host->dtype,&it->second.host->shape};}const TensorView*t=model_.find(name);if(!t)throw std::runtime_error("tensor not found: "+key);make_room(t->bytes);void*d=nullptr;check(cudaMalloc(&d,static_cast<size_t>(t->bytes)),"cudaMalloc tensor");try{check(cudaMemcpyAsync(d,t->data,static_cast<size_t>(t->bytes),cudaMemcpyHostToDevice,stream_),"upload tensor");check(cudaStreamSynchronize(stream_),"finish tensor upload");}catch(...){cudaFree(d);throw;}Entry e{d,t->bytes,1,++tick_,t};used_+=t->bytes;auto [pos,_]=entries_.emplace(std::move(key),e);return {d,e.bytes,t->dtype,&t->shape};}
void TieredStorage::release(std::string_view name) noexcept{auto it=entries_.find(std::string(name));if(it!=entries_.end()&&it->second.pins)it->second.pins--;}
void TieredStorage::clear(){check(cudaStreamSynchronize(stream_),"synchronize clear");for(auto &[_,e]:entries_)if(e.device)check(cudaFree(e.device),"cudaFree clear");entries_.clear();used_=0;}
}
