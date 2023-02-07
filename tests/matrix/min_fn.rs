use custos::{CPU, Buffer, range};
use sliced::Matrix;


#[test]
fn test_min_fn() {
    let device = CPU::new();
    
    let mut x = Matrix::from((&device, 1, 7, [10f32, -10., 10., -5., 6., 3., 1.,]));

    for i in range(100) {

        x.grad().clear();

        // add powi
        let squared = x.squared();
        
        println!("i: {:?}, sum: {:?}", i, squared.iter().sum::<f32>());

        squared.backward();

        let mut grad = x.grad();
        rawsliced::ew_assign_scalar(&mut grad, &0.1, |x, r| *x *= r);
        rawsliced::ew_assign_binary(&mut x, &grad, |x, y| *x -= y);
    }
}