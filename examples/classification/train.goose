// Full-batch 2->3 softmax classifier with deterministic synthetic data.
import deep_goose from "../../src";
import os;
fn main() {
    var x: f32[180] = [0.0; 180];
    var y: i64[90] = [0; 90];
    for row in 0..90 {
        let label = row / 30;
        y[row] = label;
        let cx: f32 = if label == 0 { -1.0 } else if label == 1 { 1.0 } else { 0.0 };
        let cy: f32 = if label == 2 { 1.0 } else { -1.0 };
        x[row*2] = cx + (((row*7)%11) as f32 - 5.0)*0.015;
        x[row*2+1] = cy + (((row*13)%17) as f32 - 8.0)*0.01;
    }
    var w: f32[6] = [0.0; 6];
    var b: f32[3] = [0.0; 3];
    var logits: f32[270] = [0.0; 270];
    var dy: f32[270] = [0.0; 270];
    var dx: f32[180] = [0.0; 180];
    var dw: f32[6] = [0.0; 6];
    var db: f32[3] = [0.0; 3];
    let start = clock_ns();
    for epoch in 0..300 {
        deep_goose::dense_forward_f32_into(x,w,b,logits,90,2,3);
        deep_goose::cross_entropy_logits_backward_f32_into(logits,y,dy,90,3);
        deep_goose::dense_backward_f32_into(x,w,dy,dx,dw,db,90,2,3);
        deep_goose::sgd_step_f32(w,dw,0.2);
        deep_goose::sgd_step_f32(b,db,0.2);
    }
    let elapsed = clock_ns() - start;
    deep_goose::dense_forward_f32_into(x,w,b,logits,90,2,3);
    let loss = deep_goose::cross_entropy_logits_mean_f32(logits,y,90,3);
    var correct: i64 = 0;
    for row in 0..90 {
        let start_index = row*3;
        var largest = logits[start_index];
        var action: i64 = 0;
        for c in 1..3 {
            if logits[start_index+c] > largest {
                largest = logits[start_index+c];
                action = c;
            }
        }
        if action == y[row] { correct += 1; }
    }
    assert(correct == 90 && loss < 0.05);
    print("CLASSIFICATION_RESULT,goose,300,",elapsed,",",loss,",",correct,",90");
}
