import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

from gym.envs.registration import register

# from load_balance.job import Job
# from load_balance.job_generator import generate_job, generate_jobs
# from load_balance.server import Server
# from load_balance.timeline import Timeline
# from load_balance.wall_time import WallTime

from tf_agents.environments.load_balance.job import Job
from tf_agents.environments.load_balance.job_generator import generate_job, generate_jobs
from tf_agents.environments.load_balance.server import Server
from tf_agents.environments.load_balance.timeline import Timeline
from tf_agents.environments.load_balance.wall_time import WallTime

class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range

    def unnormalize(self, obs):
        return (obs * self.range) + self.shift


class LoadBalanceEnv(gym.Env):
    """
    Balance the load among n (default 10) heterogeneous servers
    to minimize  average job processing time (queuing delay).
    Jobs arrive according to a Poisson process and the job size
    distributes according to a Pareto distribution.

    * STATE *
        Current Load (total work waiting in the queue +
        remaining work currently being processed (if any) among n
        servers) and the incoming job size.
        The state is represented as a vector:
        [load_server_1, load_server_2, ..., load_server_n, job_size] + [job_arrival_time] if add_time = True

    * ACTIONS *
        Which server to assign the incoming job, represented as an
        integer in [0, n-1].

    * REWARD *
        Negative time elapsed for each job in the system since last action.
        For example, the virtual time was 0 for the last action, 4 jobs
        was in the system (either in the queue waiting or being processed),
        job 1 finished at time 1, job 2 finished at time 2.4 and job 3 and 4
        are still running at the next action. The next action is taken at
        time 5. Then the reward is - (1 * 1 + 1 * 2.4 + 2 * 5).
        Thus, the sum of the rewards would be negative of total
        (waiting + processing) time for all jobs.

    * REFERENCE *
        Figure 1a, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ

        Certain optimality properties of the first-come first-served discipline for g/g/s queues.
        DJ Daley.
        Stochastic Processes and their Applications, 25:301–308, 1987.
    """

    def __init__(self, num_servers=10, num_stream_jobs=1000,
                 service_rates=(0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05),
                 job_interval=55,
                 job_size_pareto_shape=1.5,
                 job_size_pareto_scale=100.0,
                 load_balance_obs_high=500000.0,
                 normalize=False,
                 add_time=False,
                 load_state=False,
                 seed=42):
        self.normalize = normalize
        self.add_time = add_time
        self.load_state = load_state
        # total number of streaming jobs (can be very large)
        self.num_stream_jobs = num_stream_jobs
        self.num_servers = num_servers
        self.job_interval = job_interval
        self.job_size_pareto_scale = job_size_pareto_scale
        self.load_balance_obs_high = load_balance_obs_high
        self.job_size_pareto_shape = job_size_pareto_shape
        self.service_rates = service_rates
        # observation and action space
        self.setup_space(self.num_servers, self.load_balance_obs_high, self.service_rates[0])
        # random seed
        self.seed(seed)
        # global timer
        self.wall_time = WallTime()
        # uses priority queue
        self.timeline = Timeline()
        # servers
        self.servers = self.initialize_servers(self.service_rates, self.num_servers)
        # current incoming job to schedule
        self.incoming_job = None
        # finished jobs (for logging at the end)
        self.finished_jobs = []
        # reset environment (generate new jobs)
        self.reset()

    def generate_job(self):
        if self.num_stream_jobs_left > 0:
            dt, size = generate_job(self.np_random, self.job_size_pareto_shape, self.job_size_pareto_scale,
                                    self.job_interval)
            t = self.wall_time.curr_time
            self.timeline.push(t + dt, size)
            self.num_stream_jobs_left -= 1

    def generate_jobs(self):
        all_t, all_size = generate_jobs(self.num_stream_jobs, self.np_random)
        for t, size in zip(all_t, all_size):
            self.timeline.push(t, size)

    def initialize(self):
        assert self.wall_time.curr_time == 0
        # generate a job
        self.generate_job()
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)  # a job arrival event
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_servers(self, service_rates, num_servers):
        servers = []
        for server_id in range(num_servers):
            server = Server(server_id, service_rates[server_id], self.wall_time)
            servers.append(server)
        return servers

    def observe(self):
        obs_arr = []
        # load on each server
        # print(len(self.servers))
        i = 0
        for server in self.servers:
            # queuing work
            load = sum(j.size for j in server.queue)
            # print("server", i, load, "JOBS", len(server.queue), [j.size for j in server.queue])
            if server.curr_job is not None:
                # remaining work currently being processed
                if not self.load_state:
                    # Original park state formulation
                    load += server.curr_job.finish_time - self.wall_time.curr_time
                else:
                    load += int((server.curr_job.finish_time - self.wall_time.curr_time) * server.service_rate)
                # print("server", i, load, "JOBS", len(server.queue) + 1, "SIZE", server.curr_job.size, "finish time", server.curr_job.finish_time)
            # if the load is larger than observation threshold
            # report a warning
            if load > self.obs_high[server.server_id]:
                logger.warn('Server ' + str(server.server_id) + ' at time ' +
                            str(self.wall_time.curr_time) + ' has load ' + str(load) +
                            ' larger than obs_high ' + str(self.obs_high[server.server_id]))
                load = self.obs_high[server.server_id]
            obs_arr.append(load)
            i += 1
        # incoming job size
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            job_idx = -1 if not self.add_time else -2
            if self.incoming_job.size > self.obs_high[job_idx]:
                logger.warn('Incoming job at time ' + str(self.wall_time.curr_time) +
                            ' has size ' + str(self.incoming_job.size) +
                            ' larger than obs_high ' + str(self.obs_high[-1]))
                obs_arr.append(self.obs_high[job_idx])
            else:
                obs_arr.append(self.incoming_job.size)

        if self.add_time:
            obs_arr.append(self.wall_time.curr_time)
        obs_arr = np.array(obs_arr)

        try:
            assert self.un_norm_observation_space.contains(obs_arr)
        except AssertionError as err:
            raise err

        return obs_arr

    def reset(self):
        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.num_stream_jobs_left = self.num_stream_jobs
        assert self.num_stream_jobs_left > 0
        self.incoming_job = None
        self.finished_jobs = []
        # initialize environment (jump to first job arrival event)
        self.initialize()
        state = self.observe()
        return state if not self.normalize else self.state_normalizer.normalize(state)

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)[0]

    def setup_space(self, num_servers, load_balance_obs_high, low_server):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * (num_servers + 1))
        self.obs_high = np.array([load_balance_obs_high] * (num_servers + 1))
        if self.add_time:
            self.obs_low = np.concatenate((self.obs_low, np.array([0])))
            # Max value for time is dependent on time taken to complete all jobs
            # Capped value for max load on a server in original park fomulation, final time
            self.obs_high = np.concatenate((self.obs_high, np.array(
                [self.load_balance_obs_high * self.job_interval * (self.num_stream_jobs) / low_server])))
        self.un_norm_observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.observation_space = self.un_norm_observation_space if not self.normalize else spaces.Box(0.0, 1.0,
                                                                                                      shape=self.un_norm_observation_space.shape,
                                                                                                      dtype=np.float64)
        self.state_normalizer = StateNormalizer(self.un_norm_observation_space)
        self.action_space = spaces.Discrete(num_servers)

    def step(self, action):

        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        # schedule job to server
        self.servers[action].schedule(self.incoming_job)
        running_job = self.servers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        # erase incoming job
        self.incoming_job = None

        # generate next job
        self.generate_job()

        # set to compute reward from this time point
        reward = 0

        while len(self.timeline) > 0:
            new_time, obj = self.timeline.pop()

            # update reward
            num_active_jobs = sum(len(w.queue) for w in self.servers)
            for server in self.servers:
                if server.curr_job is not None:
                    assert server.curr_job.finish_time >= \
                           self.wall_time.curr_time  # curr job should be valid
                    num_active_jobs += 1
            reward -= (new_time - self.wall_time.curr_time) * num_active_jobs

            # tick time
            self.wall_time.update(new_time)

            if isinstance(obj, int):  # new job arrives
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                # break to consult agent
                break

            elif isinstance(obj, Job):  # job completion on server
                job = obj
                if not np.isinf(self.num_stream_jobs_left):
                    self.finished_jobs.append(job)
                else:
                    # don't store infinite streaming
                    # TODO: stream the complete job to some file
                    if len(self.finished_jobs) > 0:
                        self.finished_jobs[-1] += 1
                    else:
                        self.finished_jobs = [1]
                if job.server.curr_job == job:
                    # server's current job is done
                    job.server.curr_job = None
                running_job = job.server.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)

            else:
                print("illegal event type")
                exit(1)
        jobs = [len(server.queue) for server in self.servers]
        done = ((len(self.timeline) == 0) and \
                self.incoming_job is None)

        state = self.observe()
        return state if not self.normalize else self.state_normalizer.normalize(state), reward, done, {
            'curr_time': self.wall_time.curr_time, 'jobs': jobs}


register(
  id='LoadBalanceDefault-v0',
  entry_point=LoadBalanceEnv,
  max_episode_steps=1001,
  reward_threshold=-500,
)

service_rates_medium = (0.5, 0.75, 1.0, 1.25, 1.5)
num_servers_medium = len(service_rates_medium)
kwargs_medium = {"add_time": True, "load_state": True, "normalize": False, "num_stream_jobs": 250,
          "service_rates": service_rates_medium, "num_servers": num_servers_medium, "job_size_pareto_scale": 45}

register(
  id='LoadBalanceMedium-v0',
  entry_point=LoadBalanceEnv,
  max_episode_steps=251,
  reward_threshold=-250,
  kwargs=kwargs_medium
)