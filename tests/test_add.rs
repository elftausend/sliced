use custos::{Buffer, CPU};
use sliced::BinaryOpsMayGrad;

#[cfg(feature = "cpu")]
#[test]
fn test_add() {
    let device = CPU::new();

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.add(&lhs, &rhs);
    assert_eq!(out.read(), [7, 9, 11, 13, 15]);

    #[cfg(feature="autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_add_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::new(0)?;

    let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let rhs = Buffer::from((&device, [6, 7, 8, 9, 10]));

    let out = device.add(&lhs, &rhs);
    assert_eq!(out.read(), [7, 9, 11, 13, 15]);

    #[cfg(feature="autograd")]
    {
        out.backward();

        let grad = lhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);

        let grad = rhs.grad();
        assert_eq!(grad.read(), [1, 1, 1, 1, 1]);
    }
    Ok(())
}
