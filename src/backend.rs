/// Trait for pluggable computation backends.
///
/// A backend controls how the inner computation loops of the inference ops
/// are scheduled. The default [`SequentialBackend`] runs everything inline
/// on the calling thread.
///
/// # Safety contract
/// `defer_job` is given a pointer-sized `arg`. The backend **must** ensure
/// all deferred closures have completed before `wait` returns and before the
/// data pointed to by `arg` is dropped or moved.
pub trait Backend {
    /// Schedule a single unit of work.
    ///
    /// Implementations may run it immediately or defer it to a thread pool.
    fn defer_job(func: fn(usize), arg: usize);

    /// Block until all jobs previously submitted via [`defer_job`] have completed.
    fn wait();

    /// Evaluates a function across a 2D grid, potentially in parallel.
    ///
    /// The default implementation distributes `ROWS * COLS` calls across the
    /// backend's worker pool via [`defer_job`]. You should not need to
    /// override this method.
    fn from_fn<T: Send, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        use core::mem::MaybeUninit;
        use core::sync::atomic::{AtomicUsize, Ordering};

        // Initialize a 2D array of MaybeUninit explicitly
        let mut output: [[MaybeUninit<T>; ROWS]; COLS] =
            core::array::from_fn(|_| core::array::from_fn(|_| MaybeUninit::uninit()));
        let out_ptr = output.as_mut_ptr() as *mut MaybeUninit<T>;

        struct Ctx<'a, F, T> {
            func: &'a F,
            out_ptr: *mut MaybeUninit<T>,
            next_job: AtomicUsize,
        }

        let ctx = Ctx {
            func: &func,
            out_ptr,
            next_job: AtomicUsize::new(0),
        };

        fn worker<F: Fn(usize, usize) -> T + Sync, T: Send, const R: usize, const C: usize>(
            arg: usize,
        ) {
            let ctx = unsafe { &*(arg as *const Ctx<'_, F, T>) };
            let total = R * C;
            loop {
                let idx = ctx.next_job.fetch_add(1, Ordering::Relaxed);
                if idx >= total {
                    break;
                }
                // Column-major: idx = col * R + row
                let row = idx % R;
                let col = idx / R;
                let val = (ctx.func)(row, col);
                unsafe {
                    ctx.out_ptr.add(idx).write(MaybeUninit::new(val));
                }
            }
        }

        let total = ROWS * COLS;
        for _ in 0..total {
            Self::defer_job(worker::<F, T, ROWS, COLS>, &ctx as *const _ as usize);
        }
        Self::wait();

        // Convert the initialized array to SMatrix ArrayStorage safely
        unsafe {
            let ptr = &output as *const _ as *const crate::buffer::Buffer2D<T, ROWS, COLS>;
            let buffer = core::ptr::read(ptr);
            // Forget the original array so its (un)initialized contents aren't dropped
            core::mem::forget(output);
            buffer
        }
    }
}

/// The default sequential backend.
///
/// Every job submitted via [`defer_job`] is run immediately on the calling thread.
pub struct SequentialBackend;

impl Backend for SequentialBackend {
    #[inline(always)]
    fn defer_job(func: fn(usize), arg: usize) {
        func(arg);
    }

    #[inline(always)]
    fn wait() {}

    // Opt to run synchronously for minimal overhead.
    #[inline(always)]
    fn from_fn<T: Send, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        use core::mem::MaybeUninit;
        let mut output: [[MaybeUninit<T>; ROWS]; COLS] =
            core::array::from_fn(|_| core::array::from_fn(|_| MaybeUninit::uninit()));

        for col in 0..COLS {
            for row in 0..ROWS {
                output[col][row] = MaybeUninit::new(func(row, col));
            }
        }

        unsafe {
            let ptr = &output as *const _ as *const crate::buffer::Buffer2D<T, ROWS, COLS>;
            let buffer = core::ptr::read(ptr);
            core::mem::forget(output);
            buffer
        }
    }
}
