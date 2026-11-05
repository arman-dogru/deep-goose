// Independently validated rectangular OpenBLAS GEMM.
import blas from "../../backends/blas";
import deep_goose from "../../src";

fn close(actual: f32, expected: f32, tol: f32) -> bool {
    let diff = if actual > expected { actual - expected } else { expected - actual };
    diff <= tol
}

fn main() {
    let a: f32[12] = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0];
    let b: f32[8] = [1.0,-1.0,2.0,0.0,-1.0,2.0,3.0,1.0];
    let expected: f32[6] = [14.0,9.0,34.0,17.0,54.0,25.0];
    var result: f32[6] = [0.0; 6];
    var native: f32[6] = [0.0; 6];
    blas::matmul_f32_into(a, b, result, 3, 4, 2);
    deep_goose::matmul_f32_into(a, b, native, 3, 4, 2);
    for i in 0..6 {
        assert(close(result[i], expected[i], 0.00001));
        assert(close(native[i], expected[i], 0.00001));
    }

    // A second matrix with a K=1 corner case.
    let x: f32[3] = [2.0,-3.0,7.0];
    let y: f32[2] = [4.0,-5.0];
    let expected2: f32[6] = [8.0,-10.0,-12.0,15.0,28.0,-35.0];
    var actual2: f32[6] = [0.0; 6];
    blas::matmul_f32_into(x, y, actual2, 3, 1, 2);
    for i in 0..6 {
        assert(close(actual2[i], expected2[i], 0.00001));
    }
    print("PASS OpenBLAS adapter rectangular and K=1 elementwise references");
}
