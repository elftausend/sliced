use custos::{prelude::CLBuffer, CDatatype, OpenCL};

pub fn cl_onehot<T: CDatatype>(
    device: &OpenCL,
    x: &CLBuffer<T>,
    out: &mut CLBuffer<T>,
    highest_class: usize,
) -> custos::Result<()> {
    let src = format!("
        __kernel void onehot(__global {dtype}* x, __global {dtype}* out, int highest_class) {{
            size_t id = get_global_id(0);
            out[id * highest_class + (size_t) x[id]] = 1;
        }}
    ", dtype = T::as_c_type_str());

    device.launch_kernel(&src, [x.len(), 0, 0], None, &[x, out, &(highest_class as i32)])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use custos::{Buffer, OpenCL};

    use crate::{cl_onehot, ops2::max};

    #[test]
    fn test_onehot_cl() -> custos::Result<()> {
        let device = OpenCL::new(0)?;

        let x = Buffer::from((&device, [0i32, 1, 4, 3]));

        let highest_class = max(&x.read_to_vec()).unwrap() + 1;

        let mut out = Buffer::<_, _>::from((&device, vec![0; highest_class as usize * x.len()]));

        cl_onehot(&device, &x, &mut out, highest_class as usize)?;

        #[rustfmt::skip]
        let expected = vec![
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 0, 1, 0,
        ];
        assert_eq!(expected, out.read());
        Ok(())
    }
}
