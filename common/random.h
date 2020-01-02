#pragma once

#include <algorithm>
#include <memory>
#include <random>

class UniformDistRandom {
 public:
  UniformDistRandom(double min_val, double max_val) {
    std::random_device rd;
    rand_engine_ptr_ = std::make_shared<std::mt19937>(rd());
    rand_dist_ptr_ = std::make_shared<std::uniform_real_distribution<double>>(min_val, max_val);
  }

  UniformDistRandom(long seed, double min_val, double max_val) {
    rand_engine_ptr_ = std::make_shared<std::mt19937>(seed);
    rand_dist_ptr_ = std::make_shared<std::uniform_real_distribution<double>>(min_val, max_val);
  }

  ~UniformDistRandom() = default;

  double Generate() {
    return (*rand_dist_ptr_)(*rand_engine_ptr_);
  }

 private:
  std::shared_ptr<std::mt19937> rand_engine_ptr_;
  std::shared_ptr<std::uniform_real_distribution<double>> rand_dist_ptr_;
};

class NormalDistRandom {
 public:
  NormalDistRandom(double mean, double std) {
    std::random_device rd;
    rand_engine_ptr_ = std::make_shared<std::mt19937>(rd());
    rand_dist_ptr_ = std::make_shared<std::normal_distribution<double>>(mean, std);
  }

  NormalDistRandom(long seed, double mean, double std) {
    rand_engine_ptr_ = std::make_shared<std::mt19937>(seed);
    rand_dist_ptr_ = std::make_shared<std::normal_distribution<double>>(mean, std);
  }

  ~NormalDistRandom() = default;

  double Generate() {
    return (*rand_dist_ptr_)(*rand_engine_ptr_);
  }

 private:
  std::shared_ptr<std::mt19937> rand_engine_ptr_;
  std::shared_ptr<std::normal_distribution<double>> rand_dist_ptr_;
};

class RandomShuffle {
 public:
  RandomShuffle() {
    std::random_device rd;
    rand_engine_ptr_ = std::make_shared<std::mt19937>(rd());
  }
  ~RandomShuffle() = default;

  template<typename RandomIt>
  void Shuffle(RandomIt begin, RandomIt end) {
    std::shuffle(begin, end, *rand_engine_ptr_);
  }

 private:
  std::shared_ptr<std::mt19937> rand_engine_ptr_;
};
