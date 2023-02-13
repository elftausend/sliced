
#[cfg(feature="cpu")]
#[test]
fn test_row_op_cpu() {
    use custos::{CPU, Buffer};
    use sliced::RowOpMayGrad;

    let device = CPU::new();

    // 3 x 5
    #[rustfmt::skip]
    let buf = Buffer::from((&device, [
        -1, -3, 4,  5,  2,
         2,  1, 5, -3,  2,
         2,  4, 3,  2, -1,
    ]));

    let row_add = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let out = device.add_row(3, 5, &buf, &row_add);

    #[rustfmt::skip]
    assert_eq!(out.read(), [
        0, -1, 7, 9, 7,
        3, 3, 8, 1, 7,
        3, 6, 6, 6, 4
    ]);
    
    #[cfg(feature="autograd")]
    {
        out.backward();

        let row_add_grad = row_add.grad();
        assert_eq!(row_add_grad.read(), [3, 3, 3, 3, 3]);

        let buf_grad = buf.grad();
        assert_eq!(buf_grad.read(), [1; 3*5])
    }
}

#[cfg(feature="cpu")]
#[test]
fn test_row_op_mut_cpu() {
    use custos::{CPU, Buffer};
    use sliced::RowOpMayGrad;

    let device = CPU::new();

    // 3 x 5
    #[rustfmt::skip]
    let mut buf = Buffer::from((&device, [
        -1, -3, 4,  5,  2,
         2,  1, 5, -3,  2,
         2,  4, 3,  2, -1,
    ]));

    let row_add = Buffer::from((&device, [1, 2, 3, 4, 5]));

    device.add_row_mut(3, 5, &mut buf, &row_add);

    #[rustfmt::skip]
    assert_eq!(buf.read(), [
        0, -1, 7, 9, 7,
        3, 3, 8, 1, 7,
        3, 6, 6, 6, 4
    ]);

    #[cfg(feature="autograd")]
    {
        buf.backward();

        let row_add_grad = row_add.grad();
        assert_eq!(row_add_grad.read(), [3, 3, 3, 3, 3]);

        let buf_grad = buf.grad();
        assert_eq!(buf_grad.read(), [1; 3*5])
    }
}


/* 
TODO: implement this
#[cfg(feature="opencl")]
#[test]
fn test_row_op_cl() -> custos::Result<()> {
    use custos::{Buffer, OpenCL};
    use sliced::RowOpMayGrad;

    let device = OpenCL::new(0)?;

    // 3 x 5
    #[rustfmt::skip]
    let buf = Buffer::from((&device, [
        -1, -3, 4,  5,  2,
         2,  1, 5, -3,  2,
         2,  4, 3,  2, -1,
    ]));

    let row_add = Buffer::from((&device, [1, 2, 3, 4, 5]));

    let out = device.add_row(3, 5, &buf, &row_add);

    #[rustfmt::skip]
    assert_eq!(out.read(), [
        0, -1, 7, 9, 7,
        3, 3, 8, 1, 7,
        3, 6, 6, 6, 4
    ]);

    out.backward();

    let row_add_grad = row_add.grad();
    assert_eq!(row_add_grad.read(), [3, 3, 3, 3, 3]);

    let buf_grad = buf.grad();
    assert_eq!(buf_grad.read(), [1; 3*5]);

    Ok(())
}*/

#[cfg(feature="opencl")]
#[test]
fn test_row_op_mut_cl() -> custos::Result<()> {
    use custos::{OpenCL, Buffer};
    use sliced::RowOpMayGrad;

    let device = OpenCL::new(0)?;

    // 3 x 5
    #[rustfmt::skip]
    let mut buf = Buffer::from((&device, [
        -1, -3, 4,  5,  2,
         2,  1, 5, -3,  2,
         2,  4, 3,  2, -1,
    ]));

    let row_add = Buffer::from((&device, [1, 2, 3, 4, 5]));

    device.add_row_mut(3, 5, &mut buf, &row_add);

    #[rustfmt::skip]
    assert_eq!(buf.read(), [
        0, -1, 7, 9, 7,
        3, 3, 8, 1, 7,
        3, 6, 6, 6, 4
    ]);

    #[cfg(feature="autograd")]
    {
        buf.backward();

        let row_add_grad = row_add.grad();
        assert_eq!(row_add_grad.read(), [3, 3, 3, 3, 3]);

        let buf_grad = buf.grad();
        assert_eq!(buf_grad.read(), [1; 3*5]);
    }
    Ok(())
}
