#pragma once

namespace qmutils {

template <size_t... Dims>
class StaticIndex {
 public:
  static constexpr size_t Dimensions = sizeof...(Dims);

  constexpr StaticIndex() {
    static_assert(total_size() <= 64,
                  "Size exceeds maximum number of orbitals (64)");
  }

  template <typename... Coords>
  [[nodiscard]] constexpr uint8_t to_orbital(Coords... coords) const {
    static_assert(sizeof...(coords) == Dimensions,
                  "Invalid number of coordinates");
    std::array<size_t, Dimensions> coordinates = {
        static_cast<size_t>(coords)...};

    for (size_t i = 0; i < Dimensions; ++i) {
      if (coordinates[i] >= m_dimensions[i]) {
        throw std::out_of_range("Coordinates out of bounds");
      }
    }

    uint8_t orbital = 0;
    size_t multiplier = 1;
    for (size_t i = 0; i < Dimensions; ++i) {
      orbital += static_cast<uint8_t>(coordinates[i] * multiplier);
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] constexpr std::array<size_t, Dimensions> from_orbital(
      uint8_t orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds");
    }

    std::array<size_t, Dimensions> coordinates;
    for (size_t i = 0; i < Dimensions; ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  constexpr const std::array<size_t, Dimensions>& dimensions() const {
    return m_dimensions;
  }

  constexpr size_t dimension(size_t i) const {
    if (i >= Dimensions) {
      throw std::out_of_range("Dimension index out of bounds");
    }
    return m_dimensions[i];
  }

  constexpr size_t size() const { return total_size(); }

 private:
  static constexpr std::array<size_t, Dimensions> m_dimensions = {Dims...};

  static constexpr size_t total_size() { return (... * Dims); }
};

class DynamicIndex {
 public:
  DynamicIndex(std::vector<size_t> dimensions)
      : m_dimensions(std::move(dimensions)) {
    if (total_size() > 64) {
      throw std::out_of_range("Size exceeds maximum number of orbitals (64)");
    }
  }

  [[nodiscard]] uint8_t to_orbital(
      const std::vector<size_t>& coordinates) const {
    if (coordinates.size() != m_dimensions.size()) {
      throw std::out_of_range("Invalid number of coordinates");
    }

    uint8_t orbital = 0;
    size_t multiplier = 1;
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
      if (coordinates[i] >= m_dimensions[i]) {
        throw std::out_of_range("Coordinates out of bounds");
      }
      orbital += static_cast<uint8_t>(coordinates[i] * multiplier);
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] std::vector<size_t> from_orbital(uint8_t orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds");
    }

    std::vector<size_t> coordinates(m_dimensions.size());
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  const std::vector<size_t>& dimensions() const { return m_dimensions; }

  size_t dimension(size_t i) const {
    if (i >= m_dimensions.size()) {
      throw std::out_of_range("Dimension is out of bounds");
    }
    return m_dimensions[i];
  }

  size_t size() const { return total_size(); }

 private:
  std::vector<size_t> m_dimensions;

  size_t total_size() const {
    size_t size = 1;
    for (size_t dim : m_dimensions) {
      size *= dim;
    }
    return size;
  }
};

}  // namespace qmutils
