use custos::{Buffer, CPU};
use sliced::BinaryOpsMayGrad;

#[cfg(feature = "cpu")]
#[test]
fn test_sub() {
    let device = CPU::<custos::Base>::new();

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.sub(&lhs, &rhs);
    assert_eq!(out.read(), [-5, -5, -5, -5, -5]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [-1, -1, -1, -1, -1]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_sub_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::<custos::Base>::new(0)?;

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.sub(&lhs, &rhs);
    assert_eq!(out.read(), [-5, -5, -5, -5, -5]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [-1, -1, -1, -1, -1]);
    }
    Ok(())
}
