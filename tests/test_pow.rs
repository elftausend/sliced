use custos::{Buffer, CPU};
use sliced::PowMayGrad;

#[cfg(feature = "cpu")]
#[test]
fn test_pow() {
    let device = CPU::new();

    let x = Buffer::from((&device, [1., 2., 3., 4., 5.]));

    let out = device.pow(&x, 3.);

    assert_eq!(&*out, [1., 8., 27., 64., 125.,]);

    out.backward();
    assert_eq!(&*x.grad(), [3., 12., 27., 48., 75.]);
}
