#pragma once

#include <list>
#include <unordered_map>
#include <utility>

namespace qmutils {

template <typename Key, typename Value>
class LRUCache {
 public:
  using key_type = Key;
  using value_type = Value;
  using list_iterator =
      typename std::list<std::pair<key_type, value_type>>::iterator;

  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  void put(const key_type& key, const value_type& value) {
    auto it = m_cache_map.find(key);
    if (it != m_cache_map.end()) {
      m_cache_list.splice(m_cache_list.begin(), m_cache_list, it->second);
      it->second->second = value;
    } else {
      if (m_cache_list.size() >= capacity_) {
        auto last = m_cache_list.back();
        m_cache_map.erase(last.first);
        m_cache_list.pop_back();
      }
      m_cache_list.emplace_front(key, value);
      m_cache_map[key] = m_cache_list.begin();
    }
  }

  bool get(const key_type& key, value_type& value) {
    auto it = m_cache_map.find(key);
    if (it == m_cache_map.end()) {
      return false;
    }
    m_cache_list.splice(m_cache_list.begin(), m_cache_list, it->second);
    value = it->second->second;
    return true;
  }

  void clear() {
    m_cache_map.clear();
    m_cache_list.clear();
  }

  size_t size() const { return m_cache_map.size(); }

  size_t capacity() const { return capacity_; }

 private:
  size_t capacity_;
  std::list<std::pair<key_type, value_type>> m_cache_list;
  std::unordered_map<key_type, list_iterator> m_cache_map;
};

}  // namespace qmutils
