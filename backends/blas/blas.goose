// Optional OpenBLAS backend.
// The Goose-facing API checks shape and delegates row-major f32 GEMM through C.
// The native adapter does not retain any borrowed pointers.
namespace blas;

extern fn native_sgemm(a: const f32&, b: const f32&, c: f32&,
                       m: i64, k: i64, n: i64) -> i32 = "deep_goose_blas_sgemm";

fn matmul_f32_into(a: const f32[:], b: const f32[:], c: f32[:],
                   m: i64, k: i64, n: i64) {
    assert(m > 0 && k > 0 && n > 0);
    assert(m <= 1048576 && k <= 1048576 && n <= 1048576);
    assert(a.len == m*k && b.len == k*n && c.len == m*n);
    let status = native_sgemm(a[0], b[0], c[0], m, k, n);
    assert(status == 0);
}
