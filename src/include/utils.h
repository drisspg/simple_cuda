
template <typename T, typename Y> T ceil_div(T a, Y b) {
  return a / b + (a % b != 0);
}