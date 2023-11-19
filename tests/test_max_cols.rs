#[cfg(feature = "cpu")]
#[test]
fn test_max_cols_cpu() {
    use sliced::{BinaryOpsMayGrad, Buffer, MaxColsMayGrad, CPU};

    let device = CPU::<custos::Base>::new();
    let rhs = Buffer::from((&device, [1, 4, 2]));

    #[rustfmt::skip]
    let lhs = Buffer::from((&device, 
    [
        -3, 2, 3, 1,
        1, 5, -5, 4,
        -9, -2, -4, -1
    ]));

    let max_cols = device.max_cols(3, 4, &lhs);
    assert_eq!(&**max_cols, [3, 5, -1]);

    let _out = device.add(&max_cols, &rhs);

    #[cfg(feature = "autograd")]
    {
        _out.backward();

        #[rustfmt::skip]
        let expected = [
            0, 0, 1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1
        ];

        assert_eq!(&***lhs.grad(), expected);
        assert_eq!([1, 1, 1], &***rhs.grad());
    }
}
