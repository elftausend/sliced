#[cfg(feature = "cpu")]
#[cfg_attr(miri, ignore)]
#[test]
fn test_gemm_cpu() {
    use custos::{Buffer, CPU};
    use sliced::GemmMayGrad;

    let device = CPU::<custos::Base>::new();

    let m = 4;
    let k = 2;
    let n = 3;

    // 4 x 2
    #[rustfmt::skip]
    let lhs = Buffer::from((&device, [
        1., 2., 
        3., 4.,
        4., 5., 
        6., 5.
    ]));

    // 2 x 3
    let rhs = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));

    let out: Buffer = device.gemm(m, k, n, &lhs, &rhs);

    assert_eq!(
        &*out,
        [9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 24.0, 33.0, 42.0, 26.0, 37.0, 48.0]
    );

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let lhs_grad = lhs.grad();
        assert_eq!(&**lhs_grad, [6.0, 15.0, 6.0, 15.0, 6.0, 15.0, 6.0, 15.0]);

        let rhs_grad = rhs.grad();
        assert_eq!(&**rhs_grad, [14.0, 14.0, 14.0, 16.0, 16.0, 16.0]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_gemm_cl() -> custos::Result<()> {
    use custos::{Buffer, OpenCL};
    use sliced::GemmMayGrad;

    let device = OpenCL::<custos::Base>::new(0).unwrap();

    let m = 4;
    let k = 2;
    let n = 3;

    // 4 x 2
    #[rustfmt::skip]
    let lhs = Buffer::from((&device, [
        1., 2., 
        3., 4.,
        4., 5., 
        6., 5.
    ]));

    // 2 x 3
    let rhs = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));

    let out: Buffer<_, OpenCL> = device.gemm(m, k, n, &lhs, &rhs);
    assert_eq!(
        out.read(),
        [9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 24.0, 33.0, 42.0, 26.0, 37.0, 48.0]
    );

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let lhs_grad = lhs.grad();
        assert_eq!(
            lhs_grad.read(),
            [6.0, 15.0, 6.0, 15.0, 6.0, 15.0, 6.0, 15.0]
        );

        let rhs_grad = rhs.grad();
        assert_eq!(rhs_grad.read(), [14.0, 14.0, 14.0, 16.0, 16.0, 16.0]);
    }

    Ok(())
}
