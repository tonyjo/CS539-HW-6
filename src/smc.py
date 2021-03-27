import sys
import json
import time
import math
import torch
import numpy as np

from daphne import daphne
from evaluator import evaluate
# VIZ
import matplotlib.pyplot as plt


def _totensor(x):
    dtype=torch.float32 # Hard coding to float for now
    if not torch.is_tensor(x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=dtype)
        else:
            x = torch.tensor([x], dtype=dtype)
    return x


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    # uses the log_sum_exp trick to avoid overflow (i.e. subtract the max before exponentiating)
    log_weights_ = [W_l.item() for W_l in log_weights]
    max_log_wts  = max(log_weights_)
    exp_weights  = [math.exp(W_l - max_log_wts) for W_l in log_weights_]
    sum_exp_wts  = sum(exp_weights)
    norm_exp_wts = [e_l/sum_exp_wts for e_l in exp_weights]
    norm_exp_wts = _totensor(norm_exp_wts)
    if DEBUG:
        print('W: ', norm_exp_wts)
    new_samples = torch.multinomial(input=norm_exp_wts, num_samples=norm_exp_wts.shape[0], replacement=True)
    new_samples = new_samples.tolist()
    if DEBUG:
        print('S: ', new_samples)
    new_particles = [particles[int(sample)] for sample in new_samples]
    log_weights = _totensor(log_weights)
    logZ = torch.mean(log_weights)

    return logZ, new_particles


DEBUG=False
def SMC(n_particles, exp):
    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):
        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.
        particles.append(res)
        weights.append(logW)

    # Can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                # Check particle addresses
                if i == 0:
                    addr0 = res[2]['addr']
                else:
                    addri = res[2]['addr']
                    if DEBUG:
                        print('Address: ', addr0, addri)
                    if not addr0 == addri:
                        raise AssertionError(f'Address mismatch!')
                # get weights and continuations
                particles[i] = res
                weights[i] += res[2]['logProb']

        if not done:
            # Resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-6/src'

    cache = './jsons'
    use_cache = True

    for i in range(1,2):
    #for i in range(1,5):
        if use_cache:
            with open(f'{cache}/{i}.json','r') as f:
                exp = json.load(f)
        else:
            exp = daphne(['desugar-hoppl-cps', '-i', f'{program_path}/programs/{i}.daphne'])
            with open(cache + f'/{i}.json', 'w') as f:
                json.dump(exp, f)

        # output = lambda x: x # The output is the identity
        # res =  evaluate(exp, env=None)('addr_start', output) # Set up the initial call, every evaluate returns a continuation, a set of arguments, and a map sigma at every procedure call, every sample, and every observe
        # cont, args, sigma = res
        # print(cont, args, sigma)
        # # you can keep calling this to run the program forward:
        # res = cont(*args)
        # print(res)

        n_particles_runs = [1, 10, 100, 1000, 10000, 100000]
        #n_particles_runs = [1]
        #-----------------------------------------------------------------------
        # 3
        if i == 1:
            for n_particles in n_particles_runs:
                begin = time.time()
                logZ, particles = SMC(n_particles, exp)
                end = time.time()

                # Time
                hours, rem = divmod(end-begin, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Wall-clock Runtime for Particle Size {n_particles} (HR:MIN): ")
                print("{:0>2}:{:0>2}".format(int(hours), int(minutes)))
                print("--------------------------------")
                print("\n")

                # Mean:
                EX = 0.0
                for p in particles:
                    EX += p[0].item()
                EX = EX/len(particles)
                print(f"Posterior mean for {n_particles} particles: {EX}")
                print("\n")

                # Plot
                plt.hist([p[0].item() for p in particles])
                plt.savefig(f'plots/P_{i}_{n_particles}.png')
                plt.clf()
        #-----------------------------------------------------------------------
        # 2
        elif i == 2:
            for n_particles in n_particles_runs:
                begin = time.time()
                logZ, particles = SMC(n_particles, exp)
                end = time.time()

                # Time
                hours, rem = divmod(end-begin, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Wall-clock Runtime for Particle Size {n_particles} (HR:MIN): ")
                print("{:0>2}:{:0>2}".format(int(hours), int(minutes)))
                print("--------------------------------")
                print("\n")

                # Mean:
                EX = 0.0
                for p in particles:
                    EX += p[0].item()
                EX = EX/len(particles)
                print(f"Posterior mean for {n_particles} particles: {EX}")
                print("\n")

                EX2 = 0.0
                for p in particles:
                    EX2 += (p[0].item())**2
                EX2 = EX2/len(particles)
                var = EX2 - (EX**2)
                print(f"Posterior Variance for {n_particles} particles: {var}")
                print("--------------------------------")
                print("\n")

                # Plot
                plt.hist([p[0].item() for p in particles])
                plt.savefig(f'plots/P_{i}_{n_particles}.png')
                plt.clf()
        #-----------------------------------------------------------------------
        # 3
        elif i == 3:
            for n_particles in n_particles_runs:
                begin = time.time()
                logZ, particles = SMC(n_particles, exp)
                end = time.time()

                # Time
                hours, rem = divmod(end-begin, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Wall-clock Runtime for Particle Size {n_particles} (HR:MIN): ")
                print("{:0>2}:{:0>2}".format(int(hours), int(minutes)))
                print("--------------------------------")
                print("\n")

                k = 0
                samples = []
                for p in particles:
                    if k == 0:
                        samples, sig  = next(stream)
                        samples = samples.unsqueeze(0)
                        # print(samples.shape)
                    else:
                        sample, sig   = next(stream)
                        sample  = sample.unsqueeze(0)
                        samples = torch.cat((samples, sample), dim=0)
                    if k%1000 == 0:
                        print(f'Progress: {k}/{len(particles)}')
                    k += 1
                print("Posterior Mean: \n", torch.mean(samples, dim=0))
                print("\n")

                var = torch.var(samples, dim=0, unbiased=True)
                print(f"Posterior Variance for {n_particles} particles: {var}")
                print("--------------------------------")
                print("\n")

                # Plot
                fig, axs = plt.subplots(3,6)
                png = [axs[i//6,i%6].hist([a[i] for a in samples]) for i in range(17)]
                plt.tight_layout()
                plt.savefig(f'plots/P_{i}_{n_particles}.png')
        #-----------------------------------------------------------------------
        # 4
        elif i == 4:
            for n_particles in n_particles_runs:
                begin = time.time()
                logZ, particles = SMC(n_particles, exp)
                end = time.time()

                # Time
                hours, rem = divmod(end-begin, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Wall-clock Runtime for Particle Size {n_particles} (HR:MIN): ")
                print("{:0>2}:{:0>2}".format(int(hours), int(minutes)))
                print("--------------------------------")
                print("\n")

                # Mean:
                EX = 0.0
                for p in particles:
                    EX += p[0].item()
                EX = EX/len(particles)
                print(f"Posterior mean for {n_particles} particles: {EX}")
                print("\n")

                EX2 = 0.0
                for p in particles:
                    EX2 += (p[0].item())**2
                EX2 = EX2/len(particles)
                var = EX2 - (EX**2)
                print(f"Posterior Variance for {n_particles} particles: {var}")
                print("--------------------------------")
                print("\n")

                # Plot
                plt.hist([p[0].item() for p in particles])
                plt.savefig(f'plots/P_{i}_{n_particles}.png')
                plt.clf()
        #-----------------------------------------------------------------------
