use custos::{Buffer, CPU};
use sliced::BinaryOpsMayGrad;

#[cfg(feature = "cpu")]
#[test]
fn test_mul() {
    let device = CPU::<custos::Autograd<custos::Base>>::new();

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.mul(&lhs, &rhs);
    assert_eq!(out.read(), [6, 14, 24, 36, 50]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [6, 7, 8, 9, 10]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [1, 2, 3, 4, 5]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_mul_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.mul(&lhs, &rhs);
    assert_eq!(out.read(), [6, 14, 24, 36, 50]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [6, 7, 8, 9, 10]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [1, 2, 3, 4, 5]);
    }

    Ok(())
}
