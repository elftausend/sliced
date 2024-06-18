#[cfg(feature = "cpu")]
#[test]
fn test_mean_rows_cpu() {
    use sliced::{BinaryOpsMayGrad, Buffer, MeanRowsMayGrad, CPU};

    let device = CPU::<custos::Autograd<custos::Base>>::new();

    let lhs = Buffer::from((&device, [2., 3., 4.]));

    #[rustfmt::skip]
    let to_mean_rows = Buffer::from((&device, [
        1., 4., 3.,
        2., 3., 5.,
        2., 1., 6.,       
        -2., 1., 4.,
    ]));

    let mean_rows: Buffer<_, _> = device.mean_rows(3, &to_mean_rows);

    let _out = device.sub(&lhs, &mean_rows);

    #[cfg(feature = "autograd")]
    {
        _out.backward();

        assert_eq!(
            [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
            &***to_mean_rows.grad()
        );

        assert_eq!([1., 1., 1.], &***lhs.grad());
    }
}
