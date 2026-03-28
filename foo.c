// build-rel/bin/clang --target=riscv64 -march=rv64gcv -O3 -S foo.c -o foo.s
int foo(float *arr, float max_value) {
  int max_idx = 0;
  for (int i = 1; i < 128; ++i) {
    if (arr[i] == max_value) {
      max_idx = i;
      break;
    }
  }
  return max_idx;
}
