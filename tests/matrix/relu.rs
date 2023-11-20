#[cfg(feature = "cpu")]
#[test]
fn test_relu_cpu() {
    use custos::CPU;
    use sliced::Matrix;

    let device = CPU::<custos::Base>::new();

    let buf = Matrix::from((&device, 1, 5, [-1., -3., 2., 5., -1.3]));
    let out = buf.relu();

    assert_eq!(out.read(), [0., 0., 2., 5., 0.,]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = buf.grad();

        assert_eq!(grad.read(), [0., 0., 1., 1., 0.,]);
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_relu_cpu_perf() {
    use std::time::Instant;

    use custos::{Autograd, CPU};
    use sliced::Matrix;

    let device = CPU::<Autograd<custos::Base>>::new();

    let buf = Matrix::from((&device, 1, 50000, [-1., -3., 2., 5., -1.3].repeat(10000)));

    let should = [0., 0., 2., 5., 0.].repeat(10000);

    let start = Instant::now();

    for _ in 0..10000 {
        let out = buf.relu();
        out.backward();
        assert_eq!(out.read()[..5], should[..5]);
    }

    println!("elapsed: {:?}", start.elapsed());

    let out = buf.relu();

    assert_eq!(out.read(), should);

    // #[cfg(feature = "autograd")]
    // {
    //     out.backward();

    //     let grad = buf.grad();

    //     assert_eq!(grad.read(), [0., 0., 1., 1., 0.,]);
    // }
}

#[cfg(feature = "opencl")]
#[test]
fn test_relu_cl() -> custos::Result<()> {
    use custos::OpenCL;
    use sliced::Matrix;

    let device = OpenCL::<custos::Base>::new(0)?;

    let buf = Matrix::from((&device, 1, 5, [-1., -3., 2., 5., -1.3]));
    let out = buf.relu();

    assert_eq!(out.read(), [0., 0., 2., 5., 0.,]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = buf.grad();

        assert_eq!(grad.read(), [0., 0., 1., 1., 0.,]);
    }

    Ok(())
}
