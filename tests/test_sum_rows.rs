#[test]
fn test_sum_rows() {
    use sliced::{BinaryOpsMayGrad, Buffer, SumRowsMayGrad, CPU};

    let device = CPU::<custos::Base>::new();
    let rhs = Buffer::from((&device, [1, 4, 2, 3]));

    #[rustfmt::skip]
    let to_sum_rows = Buffer::from((&device, [
        4,2,1,3,
        6,1,3,1,
        5,4,1,1,
    ]));

    let x: Buffer<_> = device.sum_rows(4, &to_sum_rows);
    let _out = device.mul(&x, &rhs);

    #[cfg(feature = "autograd")]
    {
        _out.backward();

        #[rustfmt::skip]
        let expected = [
            1, 4, 2, 3,
            1, 4, 2, 3,
            1, 4, 2, 3,
        ];

        assert_eq!(&***to_sum_rows.grad(), expected);
        assert_eq!(&***rhs.grad(), [15, 7, 5, 5]);
    }
}
