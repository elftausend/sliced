use custos::{Buffer, CPU};
use sliced::SquareMayGrad;

#[cfg(feature = "cpu")]
#[test]
fn test_square() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let out = device.square(&buf);
    assert_eq!(out.read(), [1, 4, 9, 16, 25]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = buf.grad();
        assert_eq!(grad.read(), [2, 4, 6, 8, 10]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_square_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::new(0)?;

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let out = device.square(&buf);
    assert_eq!(out.read(), [1, 4, 9, 16, 25]);

    #[cfg(feature = "autograd")]
    {
        out.backward();

        let grad = buf.grad();
        assert_eq!(grad.read(), [2, 4, 6, 8, 10]);
    }
    Ok(())
}
