#[cfg(feature = "cpu")]
#[test]
fn test_transpose_cpu() {
    use custos::{Buffer, CPU};
    use sliced::Transpose;

    let device = CPU::<custos::Base>::new();

    // 2 x 3
    let x = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    let out: Buffer<i32, _> = device.transpose(2, 3, &x);
    assert_eq!(&**out, [1, 4, 2, 5, 3, 6]);

    /*let y = Buffer::from((&device, [-2, 3, -4, -2, -3, -1]));


    #[cfg(feature = "autograd")]
    {
        out.backward();
        assert_eq!(&**x.grad(), [1, 1, 1, 1, 1, 1]);
    }*/
}

#[cfg(feature = "opencl")]
#[test]
fn test_transpose_cl() -> custos::Result<()> {
    use custos::{Buffer, OpenCL};
    use sliced::Transpose;

    let device = OpenCL::<custos::Base>::new(0)?;

    // 2 x 3
    let x = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    let out: Buffer<_, _> = device.transpose(2, 3, &x);
    assert_eq!(out.read(), [1, 4, 2, 5, 3, 6]);

    Ok(())
}
