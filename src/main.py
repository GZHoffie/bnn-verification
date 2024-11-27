import numpy as np
from itertools import product
import math    
import matplotlib.pyplot as plt


class binary_function:
    def __init__(self, f, n):
        """
        parameters:
        - f: A function that takes a list of binary inputs and returns a binary output.
        - n: Number of input variables.
        """
        self.f = f
        self.n = n

    def __call__(self, x):
        return self.f(x)

    def _all_possible_inputs(self):
        res = []
        for i in range(2 ** self.n):
            res.append(list(map(int, bin(i)[2:].zfill(self.n))))

        res = np.array(res)
        
        # map (0 and 1) to (1 and -1)
        res[res == 1] = -1
        res[res == 0] = 1

        return res
    
    """
    Utilities for naive Brute force computation of Fourier coefficients.
    """
    
    def sample_input(self, num_samples, restrictions=None):
        """
        Sample a set of inputs with or without restrictions.

        parameters:
        - num_samples: Number of samples to generate.
        - restrictions: None or a list of length `self.n`. It contains either -1, 0, or 1.
                        0 means no restriction on the corresponding variable.
        """
        res = []
        for _ in range(num_samples):
            res.append(np.random.choice([1, -1], self.n))

        res = np.array(res)

        # Apply restrictions
        if restrictions is not None:
            for i in range(self.n):
                if restrictions[i] != 0:
                    res[:, i] = restrictions[i]
        
        return res
    
    

    def compute_function_values(self, inputs):
        """
        Compute all function values for the 2^n possible inputs.
        """
        f_values = np.zeros(len(inputs))
        for i in range(len(inputs)):
            f_values[i] = self.f(inputs[i])

        return f_values
    
    def compute_chi_S(self, inputs, S):
        """
        Compute the value of chi_S(x) for all possible inputs.
        """
        chi_S = np.ones(len(inputs))

        for i in S:
            chi_S *= inputs[:, i]

        return chi_S
    
    def _compute_single_fourier_coefficient(self, f_values, f_inputs, S):
        """
        Compute the exact Fourier coefficient of the function for a given subset S.
        parameters:
        - f_values: function values for all possible inputs
        - f_inputs: A list of binary inputs, e.g. [0, 1, 1] for x1 = 1, x2 = -1, x3 = -1.
        - S: A list indicating the subset of variables, e.g. [0, 2] for the x1x3 term.
        """
        # Compute values of chi_S(x) for all possible inputs.
        assert len(f_values) == len(f_inputs), "The number of inputs and values should be the same."
        chi_S = self.compute_chi_S(f_inputs, S)

        #print(chi_S)

        coefficient = np.dot(f_values, chi_S) / len(f_values)
        return coefficient
    

    def _compute_fourier_coefficients(self, f_values, f_inputs):
        """
        Return a map that maps each subset of variables to its Fourier coefficient.
        """
        coeffs = {}
        for i in range(2 ** self.n):
            subset = tuple(j for j, bit in enumerate(bin(i)[2:].zfill(self.n)) if bit == '1')
            coeffs[subset] = self._compute_single_fourier_coefficient(f_values, f_inputs, subset)
        return coeffs
    

    def fourier_exact(self):
        """
        Compute the exact Fourier coefficients.
        """
        inputs = self._all_possible_inputs()
        values = self.compute_function_values(inputs)
        return self._compute_fourier_coefficients(values, inputs)
    
    def fourier_approximate(self, num_samples):
        """
        Compute the approximate Fourier coefficients.
        """
        #if num_samples > 2 ** self.n:
        #    print("Warning: The number of samples is greater than the number of possible inputs. Consider using the exact method.")
        
        
        inputs = self.sample_input(num_samples)
        values = self.compute_function_values(inputs)
        return self._compute_fourier_coefficients(values, inputs)
    
    def fourier_approximate_S(self, num_samples, S):
        inputs = self.sample_input(num_samples)
        values = self.compute_function_values(inputs)
        return self._compute_single_fourier_coefficient(values, inputs, S)
    

    """
    Utilities for Goldreich-Levin algorithm.
    """

    
    def goldreich_levin_sample_paired_input(self, num_samples, J=[]):
        """
        Sample a set of paired inputs for the Goldreich-Levin algorithm.
        """
        samples_1 = []
        samples_2 = []
        for _ in range(num_samples):
            sample_1 = np.random.choice([1, -1], self.n)
            sample_2 = np.random.choice([1, -1], self.n)
            # Sample z uniformly at random on the coordinates in `S`
            for i in range(self.n):
                if i not in J:
                    z_i = np.random.choice([1, -1])
                    sample_1[i] = z_i
                    sample_2[i] = z_i
            
            samples_1.append(sample_1)
            samples_2.append(sample_2)
        
        samples_1 = np.array(samples_1)
        samples_2 = np.array(samples_2)
        
        return samples_1, samples_2
    
    def goldreich_levin_estimate_weight(self, num_samples, S, J):
        s1, s2 = self.goldreich_levin_sample_paired_input(num_samples, J)

        # Compute function values
        res = self.compute_function_values(s1) * self.compute_chi_S(s1, S) * \
              self.compute_function_values(s2) * self.compute_chi_S(s2, S)
        
        return np.mean(res)


    
    def goldreich_levin_bucket(self, num_samples, k, S):
        """
        Compute the Fourier weight of the bucket
        B_k,S = {S cup T: T in {k+1, k+2, ..., n}}.
        """
        # Compute the weight W_S|{k+1,...,n}
        J = list(range(k))
        weight = self.goldreich_levin_estimate_weight(num_samples, S, J)
        return weight

    def goldreich_levin_recursive(self, num_samples, k, tau, S, estimated_coeffs):
        """
        Recursive procedure to find significant Fourier coefficients.
        """

        # Compute the weight of the bucket B_k,S
        weight = self.goldreich_levin_bucket(num_samples, k, S)
        #print(f"Weight of B_{k},{S}: {weight}")

        
        if k == self.n:
            # there is only one set in the bucket
            #print(S)
            fourier_coeff = self.fourier_approximate_S(num_samples, S)
            if abs(fourier_coeff) > tau / 2:
                estimated_coeffs[tuple(S)] = fourier_coeff
                
            return estimated_coeffs
        
        if weight < tau ** 2 / 2:
            return estimated_coeffs

        # Continue if the weight is sufficiently large.
        # Recurse on S union {k}
        S_new = S + [k]
        estimated_coeffs = self.goldreich_levin_recursive(num_samples, k+1, tau, S_new, estimated_coeffs)

        # Recurse on S
        estimated_coeffs = self.goldreich_levin_recursive(num_samples, k+1, tau, S, estimated_coeffs)
        return estimated_coeffs

    
    def goldreich_levin(self, tau, num_samples):
        """
        Implements the Goldreich-Levin algorithm to find significant Fourier coefficients.
        """
        estimated_coeffs = {}
        S = []
        return self.goldreich_levin_recursive(num_samples, 0, tau, S, estimated_coeffs)
    

    """
    Utilities for Chow parameters.
    """

    def chow_parameters(self, num_samples):
        res = []
        samples = self.sample_input(num_samples)
        values = self.compute_function_values(samples)
        res.append(self._compute_single_fourier_coefficient(values, samples, []))
        for i in range(self.n):
            res.append(self._compute_single_fourier_coefficient(values, samples, [i]))
        
        return np.array(res)


class BNN:
    def __init__(self, n, h):
        """
        Creates a simple binarized neural network with random weights.

        Parameters:
        - n: Number of input variables.
        - h: Number of neurons in the hidden layer.

        Returns:
        - weights: A dictionary containing the weights of the network.
        """
        self.weights = {}
        self.biases = {}
        self.scales = {}
        # Random weights for input to hidden layer
        self.weights['W1'] = np.random.choice([-1, 1], size=(h, n))
        self.biases['B1'] = np.random.normal(0, 1, h)
        self.scales['S1'] = np.random.normal(0, 1)

        # Random weights for hidden to output layer
        self.weights['W2'] = np.random.choice([-1, 1], size=(1, h))
        self.biases['B2'] = np.random.normal(0, 1)
        self.scales['S2'] = np.random.normal(0, 1)

        self.n = n
        self.h = h
    
    def layer1(self, x):
        """
        Compute the output of the first layer.
        """
        output = np.sign(self.scales['S1'] * (np.dot(self.weights['W1'], x) + self.biases['B1']))
        # if output contains zero, replace with -1
        output[output == 0] = -1
        return output
    
    def layer2(self, x):
        """
        Compute the output of the second layer.
        """
        if self.scales['S2'] * (np.dot(self.weights['W2'], x) + self.biases['B2']) < 0:
            return -1
        else:
            return 1
    
    def full_network(self, x):
        """
        Compute the output of the full network.
        """
        return self.layer2(self.layer1(x))
    

    def fourier_exact(self):
        """
        Compute the exact Fourier coefficients of the network.
        """
        f = binary_function(self.full_network, self.n)
        return f.fourier_exact()
    
    def fourier_approximate(self, num_samples):
        """
        Compute the approximate Fourier coefficients of the network.
        """
        f = binary_function(self.full_network, self.n)
        return f.fourier_approximate(num_samples)
    
    def goldreich_levin(self, tau, num_samples):
        """
        Compute the significant Fourier coefficients of the network.
        """
        f = binary_function(self.full_network, self.n)
        return f.goldreich_levin(tau, num_samples)
    

    def chow(self, num_samples):
        # For each output in the first layer, compute the chow parameters
        chow_l1 = []
        for i in range(self.h):
            def f(x):
                if self.scales['S1'] * (np.dot(self.weights['W1'][i], x) + self.biases['B1'][i]) < 0:
                    return -1
                else:
                    return 1
            f = binary_function(f, self.n)
            chow_l1.append(f.chow_parameters(num_samples))
        
        chow_l1 = np.array(chow_l1)
        #print(chow_l1)

        # For the output layer
        chow_l2 = []
        def f(x):
            if self.scales['S2'] * (np.dot(self.weights['W2'], x) + self.biases['B2']) < 0:
                return -1
            else:
                return 1
        f = binary_function(f, self.h)
        chow_l2.append(f.chow_parameters(num_samples))
    
        chow_l2 = np.array(chow_l2)
        #print(chow_l2)
        #print(chow_l2[:,1:])

        coefficients = chow_l2[:,1:] @ chow_l1
        coefficients[:,0] += chow_l2[:,0]
        coefficients = coefficients.flatten()

        res = {}
        for i in range(len(coefficients)):
            if i == 0:
                res[tuple([])] = coefficients[i]
            else:
                res[tuple([i-1])] = coefficients[i]
        
        return res

        


def method_time_benchmark(n_list, h, num_samples, tau):
    """
    Benchmark the time complexity of the Fourier computation methods.
    """
    import time

    exact_times = []
    approx_times = []
    goldreich_levin_times = []
    chow_times = []
    
    exact_n_list = []
    approximated_n_list = []

    for n in n_list:
        bnn = BNN(n, h)

        # Exact computation is only feasible (<80s) for n <= 16
        if n <= 16:
            exact_n_list.append(n)
            start = time.time()
            bnn.fourier_exact()
            end = time.time()
            exact_times.append(end - start)
            print(f"n = {n}, exact time = {end - start}")

        if n <= 22:
            approximated_n_list.append(n)
            start = time.time()
            bnn.fourier_approximate(num_samples)
            end = time.time()
            approx_times.append(end - start)
            print(f"n = {n}, approximate time = {end - start}")

        start = time.time()
        bnn.goldreich_levin(tau, num_samples)
        end = time.time()
        goldreich_levin_times.append(end - start)
        print(f"n = {n}, goldreich-levin time = {end - start}")

        start = time.time()
        bnn.chow(num_samples)
        end = time.time()
        chow_times.append(end - start)
        print(f"n = {n}, chow time = {end - start}")

    plt.plot(exact_n_list, exact_times, label="Exact", marker='o')
    plt.plot(approximated_n_list, approx_times, label="Approximate", marker='o')
    plt.plot(n_list, goldreich_levin_times, label="Goldreich-Levin", marker='o')
    plt.plot(n_list, chow_times, label="Chow", marker='o')
    plt.xlabel("Number of input variables")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("benchmark_time.pdf")
    
    plt.show()


def benchmark_correctness(n_list, h, num_samples, tau, threshold, num_trials):
    """
    Benchmark the correctness of the Fourier computation methods, 
    by comparing the estimated coefficients with the exact ones.
    """

    recall_list = {"approximate": [], "goldreich_levin": [], "chow": []}
    precision_list = {"approximate": [], "goldreich_levin": [], "chow": []}

    for n in n_list:
        recall = {"approximate": [], "goldreich_levin": [], "chow": []}
        precision = {"approximate": [], "goldreich_levin": [], "chow": []}

        for _ in range(num_trials):
            bnn = BNN(n, h)

            # Use exact as the ground truth
            exact_coeffs = bnn.fourier_exact()
            exact_coeffs = {k: v for k, v in exact_coeffs.items() if abs(v) > threshold}

            approximated_coeffs = bnn.fourier_approximate(num_samples)
            approximated_coeffs = {k: v for k, v in approximated_coeffs.items() if abs(v) > threshold}

            goldreich_levin_coeffs = bnn.goldreich_levin(tau, num_samples)
            goldreich_levin_coeffs = {k: v for k, v in goldreich_levin_coeffs.items() if abs(v) > threshold}

            chow_coeffs = bnn.chow(num_samples)
            chow_coeffs = {k: v for k, v in chow_coeffs.items() if abs(v) > threshold}

            # Compute the recall and precision of the methods
            recall["approximate"].append(len(exact_coeffs.keys() & approximated_coeffs.keys()) / len(exact_coeffs))
            recall["goldreich_levin"].append(len(exact_coeffs.keys() & goldreich_levin_coeffs.keys()) / len(exact_coeffs))
            recall["chow"].append(len(exact_coeffs.keys() & chow_coeffs.keys()) / len(exact_coeffs))

            precision["approximate"].append(len(exact_coeffs.keys() & approximated_coeffs.keys()) / len(approximated_coeffs))
            precision["goldreich_levin"].append(len(exact_coeffs.keys() & goldreich_levin_coeffs.keys()) / len(goldreich_levin_coeffs))
            precision["chow"].append(len(exact_coeffs.keys() & chow_coeffs.keys()) / len(chow_coeffs))
        
        recall_list["approximate"].append(np.mean(recall["approximate"]))
        recall_list["goldreich_levin"].append(np.mean(recall["goldreich_levin"]))
        recall_list["chow"].append(np.mean(recall["chow"]))

        print(f"n = {n}, approximate recall = {np.mean(recall['approximate'])}, goldreich-levin recall = {np.mean(recall['goldreich_levin'])}, chow recall = {np.mean(recall['chow'])}")

        precision_list["approximate"].append(np.mean(precision["approximate"]))
        precision_list["goldreich_levin"].append(np.mean(precision["goldreich_levin"]))
        precision_list["chow"].append(np.mean(precision["chow"]))

        print(f"n = {n}, approximate precision = {np.mean(precision['approximate'])}, goldreich-levin precision = {np.mean(precision['goldreich_levin'])}, chow precision = {np.mean(precision['chow'])}")



    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    axs[0].plot(n_list, recall_list["approximate"], label="Approximate", marker='o')
    axs[0].plot(n_list, recall_list["goldreich_levin"], label="Goldreich-Levin", marker='o')
    axs[0].plot(n_list, recall_list["chow"], label="Chow", marker='o')
    axs[0].set_xlabel("Number of input variables")
    axs[0].set_ylabel("Recall")
    axs[0].legend()
    axs[0].set_title("Recall vs Number of input variables")

    axs[1].plot(n_list, precision_list["approximate"], label="Approximate", marker='o')
    axs[1].plot(n_list, precision_list["goldreich_levin"], label="Goldreich-Levin", marker='o')
    axs[1].plot(n_list, precision_list["chow"], label="Chow", marker='o')
    axs[1].set_xlabel("Number of input variables")
    axs[1].set_ylabel("Precision")
    axs[1].legend()
    axs[1].set_title("Precision vs Number of input variables")

    plt.tight_layout()
    plt.savefig("benchmark_recall_precision.pdf")
    plt.show()


def benchmark_l2_difference(n_list, h, num_samples, tau, threshold, num_trials):
    """
    Benchmark the L2 difference of the Fourier computation methods, 
    by comparing the estimated coefficients with the exact ones.
    """
    l2_diff_list = {"approximate": [], "goldreich_levin": [], "chow": []}

    for n in n_list:
        l2_diff = {"approximate": [], "goldreich_levin": [], "chow": []}

        for _ in range(num_trials):
            bnn = BNN(n, h)

            # Use exact as the ground truth
            exact_coeffs = bnn.fourier_exact()
            exact_coeffs = {k: v for k, v in exact_coeffs.items() if abs(v) > threshold}

            approximated_coeffs = bnn.fourier_approximate(num_samples)
            approximated_coeffs = {k: v for k, v in approximated_coeffs.items() if abs(v) > threshold}

            goldreich_levin_coeffs = bnn.goldreich_levin(tau, num_samples)
            goldreich_levin_coeffs = {k: v for k, v in goldreich_levin_coeffs.items() if abs(v) > threshold}

            chow_coeffs = bnn.chow(num_samples)
            chow_coeffs = {k: v for k, v in chow_coeffs.items() if abs(v) > threshold}

            # Compute the L2 difference of the methods
            l2_diff["approximate"].append(np.linalg.norm(
                np.array(list(exact_coeffs.values())) - np.array([approximated_coeffs.get(k, 0) for k in exact_coeffs.keys()])
            ))
            l2_diff["goldreich_levin"].append(np.linalg.norm(
                np.array(list(exact_coeffs.values())) - np.array([goldreich_levin_coeffs.get(k, 0) for k in exact_coeffs.keys()])
            ))
            l2_diff["chow"].append(np.linalg.norm(
                np.array(list(exact_coeffs.values())) - np.array([chow_coeffs.get(k, 0) for k in exact_coeffs.keys()])
            ))

        l2_diff_list["approximate"].append(np.mean(l2_diff["approximate"]))
        l2_diff_list["goldreich_levin"].append(np.mean(l2_diff["goldreich_levin"]))
        l2_diff_list["chow"].append(np.mean(l2_diff["chow"]))

        print(f"n = {n}, approximate L2 difference = {np.mean(l2_diff['approximate'])}, goldreich-levin L2 difference = {np.mean(l2_diff['goldreich_levin'])}, chow L2 difference = {np.mean(l2_diff['chow'])}")

    plt.plot(n_list, l2_diff_list["approximate"], label="Approximate", marker='o')
    plt.plot(n_list, l2_diff_list["goldreich_levin"], label="Goldreich-Levin", marker='o')
    plt.plot(n_list, l2_diff_list["chow"], label="Chow", marker='o')
    plt.xlabel("Number of input variables")
    plt.ylabel("L2 Difference")
    plt.legend()
    plt.savefig("benchmark_l2_difference.pdf")
    
    plt.show()






    




if __name__ == "__main__":
    """
    
    def ltf(x):
        # linear threshold function
        if x[0] == -1 and x[1] == -1: #x[0] * 1 + x[1] * 1 + x[2] * -1 + x[3] * 1 + x[4] * 1 < 0:
            return -1
        else:
            return 1
    
    f = binary_function(ltf, 2)

    print(f.fourier_exact())

    #print(f.fourier_approximate(10000))

    #print(f.goldreich_levin(0.1, 1000))
    """

    """
    bnn = BNN(5, 3)
    print(bnn.fourier_exact())
    print(bnn.fourier_approximate(1000))
    print(bnn.goldreich_levin(0.1, 1000))
    print(bnn.chow(1000))
    """

    #method_time_benchmark(range(1, 30), 3, 1000, 0.1)
    #benchmark_correctness(range(1, 15), 3, 2000, 0.1, 0.1, 10)
    benchmark_l2_difference(range(1, 15), 3, 2000, 0.1, 0.1, 10)
   

    