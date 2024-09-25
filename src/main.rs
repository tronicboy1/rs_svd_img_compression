use std::time::Duration;

use nalgebra::DMatrix;

type ImageMatrix = DMatrix<f64>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("images/PureMichigan-Background-EbanIceCaves2_0.jpg")?;

    let grayscale_matrix = img
        .grayscale()
        //.resize(400, 400, image::imageops::Nearest)
        .to_luma8();

    let mut matrix = ImageMatrix::zeros(
        grayscale_matrix.height() as usize,
        grayscale_matrix.width() as usize,
    );

    for (row_i, row) in grayscale_matrix.rows().enumerate() {
        for (col_i, entry) in row.enumerate() {
            let mat_entry = matrix.get_mut((row_i, col_i)).expect("must have entry");

            *mat_entry = entry[0] as f64;
        }
    }

    let svd = nalgebra::SVD::new(matrix, true, true);

    let max_rank = svd.rank(1.0);

    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    let mut svd_s_diag = ImageMatrix::zeros(u.nrows(), v_t.ncols());
    svd_s_diag.set_diagonal(&svd.singular_values);

    let mut sleep_time = 500;

    for rank in 1..max_rank {
        std::thread::sleep(Duration::from_millis(sleep_time));

        if sleep_time > 150 {
            sleep_time -= 5;
        }

        let u_ncols = u.columns(0, rank);
        let nxn_sigma = svd_s_diag.view((0, 0), (rank, rank));
        let v_t_nrows = v_t.rows(0, rank);

        println!("U~ : {}x{}", u_ncols.nrows(), nxn_sigma.ncols());
        println!("SV~ : {}x{}", nxn_sigma.nrows(), nxn_sigma.ncols());
        println!("V'~ : {}x{}", v_t_nrows.nrows(), v_t_nrows.ncols());

        let img_approx_matrix = u_ncols * nxn_sigma * v_t_nrows;

        println!(
            "SVD Rank {} result {}x{}, original: {}x{}",
            rank,
            img_approx_matrix.row_iter().count(),
            img_approx_matrix.column_iter().count(),
            grayscale_matrix.height(),
            grayscale_matrix.width()
        );

        let mut img_r_1 =
            image::DynamicImage::new_luma8(grayscale_matrix.width(), grayscale_matrix.height());

        for (row_i, pixels) in img_r_1.as_mut_luma8().unwrap().rows_mut().enumerate() {
            let svd_row = img_approx_matrix.row(row_i);

            for (col_i, pixel) in pixels.enumerate() {
                let entry = svd_row.get(col_i).expect("must have entry");

                pixel[0] = *entry as u8;
            }
        }

        img_r_1
            .save(std::path::Path::new("./output/rank_1.png"))
            .unwrap();
    }

    Ok(())
}
