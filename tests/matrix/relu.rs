#[cfg(feature = "cpu")]
#[test]
fn test_relu_cpu() {
    use custos::CPU;
    use sliced::Matrix;

    let device = CPU::new();

    let buf = Matrix::from((&device, 1, 5, [-1., -3., 2., 5., -1.3]));
    let out = buf.relu();

    assert_eq!(out.read(), [0., 0., 2., 5., 0.,]);
    
    #[cfg(feature="autograd")]
    {
        out.backward();

        let grad = buf.grad();

        assert_eq!(grad.read(), [0., 0., 1., 1., 0.,]);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_relu_cl() -> custos::Result<()> {
    use custos::OpenCL;
    use sliced::Matrix;

    let device = OpenCL::new(0)?;

    let buf = Matrix::from((&device, 1, 5, [-1., -3., 2., 5., -1.3]));
    let out = buf.relu();

    assert_eq!(out.read(), [0., 0., 2., 5., 0.,]);

    #[cfg(feature="autograd")]
    {
        out.backward();

        let grad = buf.grad();

        assert_eq!(grad.read(), [0., 0., 1., 1., 0.,]);
    }

    Ok(())
}
