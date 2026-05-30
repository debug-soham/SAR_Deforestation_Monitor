// sar_core/src/lib.rs

use numpy::ndarray::Axis;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn fast_lee_filter<'py>(
    py: Python<'py>,
    input_matrix: PyReadonlyArray2<'py, f64>,
    window_size: usize,
    noise_variance: f64,
) -> Bound<'py, PyArray2<f64>> {
    // <-- Updated to the new Bound API return type
    let input = input_matrix.as_array();
    let (height, width) = (input.shape()[0], input.shape()[1]);

    let mut output = numpy::ndarray::Array2::<f64>::zeros((height, width));
    let offset = window_size / 2;

    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(y, mut out_row)| {
            if y >= offset && y < height - offset {
                for x in offset..(width - offset) {
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;
                    let mut count = 0.0;

                    for wy in -(offset as isize)..=(offset as isize) {
                        for wx in -(offset as isize)..=(offset as isize) {
                            let yy = (y as isize + wy) as usize;
                            let xx = (x as isize + wx) as usize;
                            let val = input[[yy, xx]];

                            sum += val;
                            sum_sq += val * val;
                            count += 1.0;
                        }
                    }

                    let local_mean = sum / count;
                    let local_variance = (sum_sq / count) - (local_mean * local_mean);

                    let mut weight = 0.0;
                    if local_variance > 0.0 {
                        weight = local_variance / (local_variance + noise_variance);
                    }

                    let current_pixel = input[[y, x]];
                    out_row[x] = local_mean + weight * (current_pixel - local_mean);
                }
            }
        });

    output.into_pyarray(py)
}

#[pymodule]
fn sar_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_lee_filter, m)?)?;
    Ok(())
}
