// use ndarray::{Array1, Array2};
// use ndarray_linalg::{QR, Solve};

// pub fn ols(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Vec<f64> {
//     // Convert input data to ndarray
//     let x = Array2::from_shape_vec((x.len(), x[0].len()), x.iter().flatten().copied().collect()).unwrap();
//     let y = Array1::from_shape_vec(y.len(), y.clone()).unwrap();

//     // Compute QR decomposition of x
//     //let qr = x.qr().unwrap();
//     let (q,r) = x.qr().unwrap();

//     // Compute Q'y
//     //let qy = qr.0.t().dot(&y);
//     let qy: Array1<f64> = q.t().dot(&y);

//     // Compute Q'y
//     //qy = qr.1.t().dot(&y);

//     // Solve Rx = Q'y
//     let b = r.solve_into(qy).unwrap();

//     // Return solution vector
//     //b.axis_iter(Axis(0)).map(|row| row[0]).collect()
//     b.to_vec()
// }

pub fn qr_decomposition(a: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let m = a[0].len();

    let mut q = vec![vec![0.0; m]; n];
    let mut r = vec![vec![0.0; m]; m];

    for j in 0..m {
        for i in 0..n {
            q[i][j] = a[i][j];
        }
        for k in 0..j {
            let dot_product = (0..n).fold(0.0, |acc, i| acc + q[i][j] * r[k][j]);
            for i in 0..n {
                q[i][j] -= dot_product * r[k][j];
            }
        }
        let norm = (0..n).map(|i| q[i][j] * q[i][j]).sum::<f64>().sqrt();
        for i in 0..n {
            q[i][j] /= norm;
        }
        for k in j..m {
            r[j][k] = (0..n).fold(0.0, |acc, i| acc + q[i][j] * a[i][k]);
        }
    }

    (q, r)
}



pub fn invert_r(r: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = r.len();
    let mut inv_r = vec![vec![0.0; n]; n];

    for i in (0..n).rev() {
        inv_r[i][i] = 1.0 / r[i][i];
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in (i + 1)..=j {
                s += r[i][k] * inv_r[k][j];
            }
            inv_r[i][j] = -s / r[i][i];
        }
    }

    inv_r
}


pub fn matrix_product(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut result = vec![vec![0.; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..b.len() {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

pub fn matrix_vector_product(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let mut result = Vec::with_capacity(matrix.len());
    for row in matrix.iter() {
        let mut sum = 0.0;
        for (i, val) in row.iter().enumerate() {
            sum += val * vector[i];
        }
        result.push(sum);
    }
    result
}

pub fn transpose(vec: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = vec.len();
    let m = vec[0].len();
    let mut transposed = vec![vec![0.; n]; m];
    for i in 0..m {
        for j in 0..n {
            transposed[i][j] = vec[j][i];
        }
    }
    transposed
}

pub fn transpose_in_place(matrix: &mut Vec<Vec<f64>>) {
    for i in 0..matrix.len() {
        for j in i+1..matrix.len() {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

pub fn transpose2(v: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| *n.next().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect()
}

pub fn transpose_times_matrix(matrix: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    transpose(matrix);
    matrix_product(matrix,matrix2)
    // let n_rows = matrix[0].len();
    // let n_cols = matrix.len();
    // let mut result = vec![vec![0.0; n_cols]; n_cols];

    // for i in 0..n_cols {
    //     for j in 0..n_cols {
    //         for k in 0..n_rows {
    //             result[i][j] += matrix[k][i] * matrix[k][j];
    //         }
    //     }
    // }

    // result
}

pub fn invert_matrix(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut a_augmented = augment_matrix(a);
    let mut b = create_identity_matrix(n);
    for i in 0..n {
        let pivot_row = find_pivot_row(&a_augmented, i);
        swap_rows(&mut a_augmented, i, pivot_row);
        swap_rows(&mut b, i, pivot_row);
        for j in i + 1..n {
            let factor = a_augmented[j][i] / a_augmented[i][i];
            subtract_rows(&mut a_augmented, i, j, factor);
            subtract_rows(&mut b, i, j, factor);
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            let factor = a_augmented[i][j];
            subtract_rows(&mut b, j, i, factor);
        }
        let factor = 1.0 / a_augmented[i][i];
        multiply_row(&mut b, i, factor);
    }
    b
}

pub fn augment_matrix(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut a_augmented = a.clone();
    for i in 0..n {
        a_augmented[i].extend_from_slice(&create_zero_vector(n - 1));
    }
    a_augmented
}

pub fn create_identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
    }
    matrix
}

pub fn find_pivot_row(a: &Vec<Vec<f64>>, col: usize) -> usize {
    let mut max_index = col;
    let mut max_value = a[col][col].abs();
    for i in col + 1..a.len() {
        let value = a[i][col].abs();
        if value > max_value {
            max_index = i;
            max_value = value;
        }
    }
    max_index
}

pub fn swap_rows(a: &mut Vec<Vec<f64>>, i: usize, j: usize) {
    let temp = a[i].clone();
    a[i] = a[j].clone();
    a[j] = temp;
}

pub fn subtract_rows(a: &mut Vec<Vec<f64>>, from: usize, to: usize, factor: f64) {
    for j in 0..a[0].len() {
        a[to][j] -= factor * a[from][j];
    }
}

fn multiply_row(a: &mut Vec<Vec<f64>>, row: usize, factor: f64) {
    for j in 0..a[0].len() {
        a[row][j] *= factor;
    }
}

pub fn create_zero_vector(n: usize) -> Vec<f64> {
    vec![0.0; n]
}




pub fn invert_lu(mat: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut pivots = vec![0; n];

    // perform LU decomposition with partial pivoting
    let mut sign = 1.0;
    for i in 0..n {
        let mut max = 0.0;
        let mut imax = i;
        for j in i..n {
            let abs = mat[j][i].abs();
            if abs > max {
                max = abs;
                imax = j;
            }
        }
        if imax != i {
            mat.swap(i, imax);
            sign = -sign;
        }
        pivots[i] = imax;
        for j in (i + 1)..n {
            let ratio = mat[j][i] / mat[i][i];
            mat[j][i] = ratio;
            for k in (i + 1)..n {
                mat[j][k] -= ratio * mat[i][k];
            }
        }
    }

    // solve the linear system for each column of the identity matrix
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut b = vec![0.0; n];
        b[j] = 1.0;
        for i in 0..n {
            let mut sum = b[pivots[i]];
            for k in 0..i {
                sum -= mat[i][k] * b[pivots[k]];
            }
            b[pivots[i]] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = b[pivots[i]];
            for k in (i + 1)..n {
                sum -= mat[i][k] * b[pivots[k]];
            }
            b[pivots[i]] = sum / mat[i][i];
        }
        for i in 0..n {
            inv[i][j] = b[i];
        }
    }

    inv
}




pub fn invert_lu_inplace(mat: &mut [f64], n: usize) {
    let mut pivots = vec![0; n];

    // perform LU decomposition with partial pivoting
    let mut sign = 1.0;
    for i in 0..n {
        let mut max = 0.0;
        let mut imax = i;
        for j in i..n {
            let abs = mat[j * n + i].abs();
            if abs > max {
                max = abs;
                imax = j;
            }
        }
        if imax != i {
            for j in 0..n {
                mat.swap(i * n + j, imax * n + j);
            }
            sign = -sign;
        }
        pivots[i] = imax;
        for j in (i + 1)..n {
            let ratio = mat[j * n + i] / mat[i * n + i];
            mat[j * n + i] = ratio;
            for k in (i + 1)..n {
                mat[j * n + k] -= ratio * mat[i * n + k];
            }
        }
    }

    // solve the linear system for each column of the identity matrix
    for j in 0..n {
        let mut b = vec![0.0; n];
        b[j] = 1.0;
        for i in 0..n {
            let mut sum = b[pivots[i]];
            for k in 0..i {
                sum -= mat[i * n + k] * b[pivots[k]];
            }
            b[pivots[i]] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = b[pivots[i]];
            for k in (i + 1)..n {
                sum -= mat[i * n + k] * b[pivots[k]];
            }
            mat[i * n + j] = sum / mat[i * n + i];
        }
    }
}


//matrix product storing the result in the second vectors
pub fn vec_product_inplace(v1: &Vec<f64>, v2: &mut Vec<f64>) {
    assert_eq!(v1.len(), v2.len(), "Vectors must have the same length.");

    let n = v1.len();

    for i in 0..n {
        for j in 0..n {
            v2[i*n + j] = v1[i] * v2[j];
        }
    }
}


pub fn transpose_mat(mat: &Vec<f64>, nrows: usize, ncols: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; mat.len()];
    for i in 0..nrows {
        for j in 0..ncols {
            let idx1 = i * ncols + j;
            let idx2 = j * nrows + i;
            transposed[idx2] = mat[idx1];
        }
    }
    transposed
}


pub fn mat_vec_product(mat: &Vec<f64>, vec: &Vec<f64>, nrows: usize, ncols: usize) -> Vec<f64> {
    assert_eq!(mat.len(), nrows * ncols);
    assert_eq!(vec.len(), ncols);
    let mut res = vec![0.0; nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            res[i] += mat[i * ncols + j] * vec[j];
        }
    }
    res
}


pub fn vec_product(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {

    let n = v1.len();
    let mut res = vec![0.0; n*n];

    for i in 0..n {
        for j in 0..n {
            res[i*n + j] = v1[i] * v2[j];
        }
    }

    res
}
