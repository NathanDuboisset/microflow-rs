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
        // Safety: T is always a quantized integer type (or array thereof) for which
        // zero bytes are a valid representation. Workers overwrite every cell before
        // `wait()` returns, so no zeroed value is ever observed.
        let mut output: [[T; ROWS]; COLS] = unsafe { core::mem::zeroed() };
        let output_ptr = output.as_mut_ptr() as *mut T;

        struct Job<'a, F, T> {
            func: &'a F,
            output_ptr: *mut T,
            row: usize,
        }

        let jobs: [Job<'_, F, T>; ROWS] = core::array::from_fn(|row| Job {
            func: &func,
            output_ptr,
            row,
        });

        fn worker<F, T, const R: usize, const C: usize>(argument: usize)
        where
            F: Fn(usize, usize) -> T + Sync,
            T: Send,
        {
            // Safety: `argument` is a `&Job` erased to `usize` by the caller.
            // `Backend::wait()` guarantees the worker finishes before `jobs` is dropped.
            let job = unsafe { &*(argument as *const Job<'_, F, T>) };
            for column in 0..C {
                let value = (job.func)(job.row, column);
                // Safety: each row is owned by exactly one job, so the column-major
                // index `column * R + job.row` is unique across concurrent writes
                // and no two threads alias the same address.
                unsafe {
                    *job.output_ptr.add(column * R + job.row) = value;
                }
            }
        }

        for job in &jobs {
            Self::defer_job(worker::<F, T, ROWS, COLS>, job as *const _ as usize);
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
