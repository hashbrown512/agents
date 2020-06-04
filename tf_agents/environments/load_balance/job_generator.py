
def generate_job(np_random, job_size_pareto_shape, job_size_pareto_scale, job_interval):
    # pareto distribution
    size = int((np_random.pareto(
            job_size_pareto_shape) + 1) * \
            job_size_pareto_scale)

    t = int(np_random.exponential(job_interval))
    return t, size

def generate_jobs(num_stream_jobs, np_random):

    # time and job size
    all_t = []
    all_size = []

    # generate streaming sequence
    t = 0
    for _ in range(num_stream_jobs):
        dt, size = generate_job(np_random)
        t += dt
        all_t.append(t)
        all_size.append(size)

    return all_t, all_size
