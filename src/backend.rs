/// Trait for pluggable computation backends.
///
/// A backend controls how the inner computation loops of the inference ops
/// are scheduled.

/// `defer_job` is given a pointer-sized `arg`. The backend **must** ensure
/// all deferred closures have completed before `wait` returns and before the
/// data pointed to by `arg` is dropped or moved.
pub trait Backend {
    /// Schedule a single unit of work.
    fn defer_job(func: fn(usize), arg: usize);

    /// Block until all jobs previously submitted via [`defer_job`] have completed.
    fn wait();

    /// Evaluates a function across a 2D grid, potentially in parallel.
    fn from_fn<T: Send + Copy, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        use core::sync::atomic::{AtomicUsize, Ordering};

        // Safety: T is always a quantized integer type (or array thereof) for which
        // zero bytes are a valid representation. Workers overwrite every cell before
        // `wait()` returns, so no zeroed value is ever observed.
        let mut output: [[T; ROWS]; COLS] = unsafe { core::mem::zeroed() };
        let output_ptr = output.as_mut_ptr() as *mut T;

        struct Context<'a, F, T> {
            func: &'a F,
            output_ptr: *mut T,
            next_row: AtomicUsize,
        }

        let context = Context {
            func: &func,
            output_ptr,
            next_row: AtomicUsize::new(0),
        };

        fn worker<F, T, const R: usize, const C: usize>(argument: usize)
        where
            F: Fn(usize, usize) -> T + Sync,
            T: Send,
        {
            // Safety: `argument` is a `&Context` erased to `usize` by the caller.
            // `Backend::wait()` guarantees all workers finish before `context` is dropped.
            let context = unsafe { &*(argument as *const Context<'_, F, T>) };
            loop {
                let row = context.next_row.fetch_add(1, Ordering::Relaxed);
                if row >= R {
                    break;
                }
                for column in 0..C {
                    let value = (context.func)(row, column);
                    // Safety: each `row` is claimed by exactly one worker via the atomic.
                    // The column-major index `column * R + row` is therefore unique across
                    // all concurrent writes, so no two threads alias the same address.
                    unsafe {
                        *context.output_ptr.add(column * R + row) = value;
                    }
                }
            }
        }

        for _ in 0..ROWS {
            Self::defer_job(worker::<F, T, ROWS, COLS>, &context as *const _ as usize);
        }
        Self::wait();

        crate::buffer::Buffer2D::from_array_storage(nalgebra::ArrayStorage(output))
    }
}

/// The default sequential backend.
///
/// Every job submitted is run immediately on the calling thread.
pub struct SequentialBackend;

impl Backend for SequentialBackend {
    #[inline(always)]
    fn defer_job(func: fn(usize), arg: usize) {
        func(arg);
    }

    #[inline(always)]
    fn wait() {}

    #[inline(always)]
    fn from_fn<T: Send + Copy, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        let output: [[T; ROWS]; COLS] =
            core::array::from_fn(|column| core::array::from_fn(|row| func(row, column)));
        crate::buffer::Buffer2D::from_array_storage(nalgebra::ArrayStorage(output))
    }
}
