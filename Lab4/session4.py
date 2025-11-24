from load_data import G, h
import snap
import random

#--------Taks 1-------
def get_average(network, attributes):
    return sum(attributes[key] for key in attributes) / len(attributes)




#--------Taks 2-------
def uniform_rdm_sampling(network, attributes, S): 
    nodes = [NI.GetId() for NI in network.Nodes()]
    samples = []

    for _ in range(S):
        n = random.choice(nodes)
        samples.append(h[n])

    return sum(samples) / len(samples)

#--------Taks 3-------
def rndm_neighbor(network, attributes, S):
    nodes = [NI.GetId() for NI in network.Nodes()]
    samples  =[]

    for _ in range(S):
        n_prime = random.choice(nodes)
        n_prime_node = network.GetNI(n_prime)
        degree = n_prime_node.GetDeg()
        if degree == 0:
            continue
        kn_prime = random.randint(0, degree - 1)
        n = n_prime_node.GetNbrNId(kn_prime)
        samples.append(attributes[n])
    return sum(samples) / len(samples)

#--------Taks 4-------
def rndm_walk(network, attributes, S):
    nodes = [NI.GetId() for NI in network.Nodes()]
    current = random.choice(nodes)
    #Steady-state
    for _ in range(S):
        node = network.GetNI(current)
        currents_degrees = node.GetDeg()
        next_walk = random.randint(0, currents_degrees - 1)
        current = node.GetNbrNId(next_walk)
    x_values = []
    for _ in range(S):
        node = network.GetNI(current)
        currents_degrees = node.GetDeg()
        next_walk = random.randint(0, currents_degrees - 1)
        current = node.GetNbrNId(next_walk)
        x_values.append(attributes[current])
    return sum(x_values) / len(x_values)

#--------Taks 5-------
def metroplis_hastings(network, attributes, S):
    nodes = [NI.GetId() for NI in network.Nodes()]
    current = random.choice(nodes)
    #Steady-state
    for _ in range(S):
        kn = network.GetNI(current)
        kn_degrees = kn.GetDeg()
        next_walk = random.randint(0, kn_degrees - 1)
        neighbor = kn.GetNbrNId(next_walk)
        kn_prime = network.GetNI(neighbor)
        kn_prime_degrees = kn_prime.GetDeg()
        walking_prob = min(1, kn_degrees/kn_prime_degrees)
        if random.random() <= walking_prob:
            current = neighbor
    x_values = []
    for _ in range(S):
        kn = network.GetNI(current)
        kn_degrees = kn.GetDeg()
        next_walk = random.randint(0, kn_degrees - 1)
        neighbor = kn.GetNbrNId(next_walk)
        kn_prime = network.GetNI(neighbor)
        kn_prime_degrees = kn_prime.GetDeg()
        walking_prob = min(1, kn_degrees/kn_prime_degrees)
        if random.random() <= walking_prob:
            current = neighbor
        x_values.append(attributes[current])
    return sum(x_values) / len(x_values)


def main():

    S = 10000
    true_avg = get_average(G, h)

    # Compute expected value for random neighbor of random node
    total = 0.0
    for NI in G.Nodes():
        u = NI.GetId()
        deg_u = NI.GetDeg()
        if deg_u == 0:
            continue
        # Sum h-values of neighbors
        neighbor_sum = sum(h[NI.GetNbrNId(i)] for i in range(deg_u))
        total += neighbor_sum / deg_u

    exp_neighbor = total / G.GetNodes()

    # Uniform random walk expected value (degree-weighted)
    total_deg = sum(NI.GetDeg() for NI in G.Nodes())
    exp_random_walk = sum(NI.GetDeg() / total_deg * h[NI.GetId()] for NI in G.Nodes())

    # Uniform node sampling and Metropolis-Hastings (unbiased)
    exp_uniform_node = true_avg
    exp_MH = true_avg

    print("-- expected values of <x>-hat -----")
    print(f"uniform sampling: {exp_uniform_node:.3f}")
    print(f"random connection of random node: {exp_neighbor:.3f}")
    print(f"uniform random walk: {exp_random_walk:.3f}")
    print(f"M-H random walk: {exp_MH:.3f}")

    print("---estimated <x> -----")

    # Uniform sampling
    for i in range(5):
        print(f"uniform sampling: {uniform_rdm_sampling(G, h, S):.3f}")

    # Random neighbor
    for i in range(5):
        print(f"random connection of random node: {rndm_neighbor(G, h, S):.3f}")

    # Uniform RW
    for i in range(5):
        print(f"uniform random walk: {rndm_walk(G, h, S):.3f}")

    # MH RW
    for i in range(5):
        print(f"M-H random walk: {metroplis_hastings(G, h, S):.3f}")


if __name__ == "__main__":
    main()

        



