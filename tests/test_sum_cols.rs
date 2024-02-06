#[cfg(feature = "cpu")]
#[test]
fn test_sum_cols_cpu() {
    use sliced::{BinaryOpsMayGrad, Buffer, SumColsMayGrad, CPU};

    let device = CPU::<custos::Autograd<custos::Base>>::new();
    let rhs = Buffer::from((&device, [1, 4, 2]));

    #[rustfmt::skip]
    let to_sum_cols = Buffer::from((&device, [
        4,2,1,3,
        6,1,3,1,
        5,4,1,1,
    ]));

    let x: Buffer<_> = device.sum_cols(4, &to_sum_cols);
    let _out = device.mul(&x, &rhs);

    #[cfg(feature = "autograd")]
    {
        _out.backward();

        #[rustfmt::skip]
        let expected = [
            1,1,1,1,
            4,4,4,4,
            2,2,2,2,
        ];

        assert_eq!(&***to_sum_cols.grad(), expected);
        assert_eq!([10, 11, 11], &***rhs.grad());
    }
}
