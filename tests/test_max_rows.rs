#[cfg(feature = "cpu")]
#[test]
fn test_max_rows_cpu() {
    use sliced::{BinaryOpsMayGrad, Buffer, MaxRowsMayGrad, CPU};

    let device = CPU::<custos::Base>::new();
    let rhs = Buffer::from((&device, [2, 3, 4, 1]));

    #[rustfmt::skip]
    let lhs = Buffer::from((&device, 
    [
        -3, 2, 3, 1,
        1, 5, -5, 4,
        -9, -2, -4, -1
    ]));

    let max_rows: Buffer<_, _> = device.max_rows(4, &lhs);

    let _out = device.mul(&max_rows, &rhs);

    #[cfg(feature = "autograd")]
    {
        _out.backward();

        #[rustfmt::skip]
        let expected = [
            0, 0, 4, 0,
            2, 3, 0, 1,
            0, 0, 0, 0
        ];

        assert_eq!(&***lhs.grad(), expected);
        assert_eq!(&***rhs.grad(), [1, 5, 3, 4]);
    }
}
