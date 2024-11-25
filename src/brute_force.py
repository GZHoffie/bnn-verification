import numpy as np
from itertools import product
import math


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
    
    def sample_paired_input_GL(self, num_samples, J=[]):
        """
        Sample a set of paired inputs for the Goldreich-Levin algorithm.
        """
        samples_1 = []
        samples_2 = []
        for _ in range(num_samples):
            sample_1 = np.random.choice([1, -1], self.n)
            sample_2 = np.random.choice([1, -1], self.n)
            # Sample z uniformly at random on the coordinates in `S`
            for i in J:
                z_i = np.random.choice([1, -1])
                sample_1[i] = z_i
                sample_2[i] = z_i
            
            samples_1.append(sample_1)
            samples_2.append(sample_2)
        
        samples_1 = np.array(samples_1)
        samples_2 = np.array(samples_2)
        
        return samples_1, samples_2
    
    def estimate_weight_GL(self, num_samples, S, J):
        s1, s2 = self.sample_paired_input_GL(num_samples, J)

        # Compute function values
        res = self.compute_function_values(s1) * self.compute_chi_S(s1, S) * \
              self.compute_function_values(s2) * self.compute_chi_S(s2, S)
        
        return np.mean(res)


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
        chi_S = np.ones(len(f_values))

        for i in S:
            chi_S *= f_inputs[:, i]

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
        if num_samples > 2 ** self.n:
            print("Warning: The number of samples is greater than the number of possible inputs. Consider using the exact method.")
        
        
        inputs = self.sample_input(num_samples)
        values = self.compute_function_values(inputs)
        return self._compute_fourier_coefficients(values, inputs)
    
    def fourier_approximate_S(self, num_samples, S):
        inputs = self.sample_input(num_samples)
        values = self.compute_function_values(inputs)
        return self._compute_single_fourier_coefficient(values, inputs, S)

    
    def GL_bucket(self, num_samples, k, S):
        """
        Compute the Fourier weight of the bucket
        B_k,S = {S cup T: T in {k+1, k+2, ..., n}}.
        """
        # Compute the weight W_S|{k+1,...,n}
        J = list(range(k, self.n))
        weight = self.estimate_weight_GL(num_samples, S, J)
        return weight

    def goldreich_levin_recursive(self, num_samples, k, epsilon, S, estimated_coeffs):
        """
        Recursive procedure to find significant Fourier coefficients.
        """

        # Compute the weight of the bucket B_k,S
        weight = self.GL_bucket(num_samples, k, S)
        print(f"Weight of B_{k},{S}: {weight}")

        if abs(weight) < epsilon / 2:
            return estimated_coeffs
        
        if k == self.n:
            # there is only one set in the bucket
            #print(S)
            estimated_coeffs[tuple(S)] = self.fourier_approximate_S(num_samples, S)
            return estimated_coeffs

        # Continue if the weight is sufficiently large.
        # Recurse on S union {k}
        S_new = S + [k]
        estimated_coeffs = self.goldreich_levin_recursive(num_samples, k+1, epsilon, S_new, estimated_coeffs)

        # Recurse on S
        estimated_coeffs = self.goldreich_levin_recursive(num_samples, k+1, epsilon, S, estimated_coeffs)
        return estimated_coeffs

    
    def goldreich_levin(self, epsilon, num_samples):
        """
        Implements the Goldreich-Levin algorithm to find significant Fourier coefficients.

        Parameters:
        - n: Number of input variables.
        - bnn_function: Function that takes a binary vector x and returns -1 or 1.
        - epsilon: Threshold for significant Fourier coefficients.

        Returns:
        - significant_S: List of subsets S where |hat{f}(S)| >= epsilon/2.
        - estimated_coeffs: Dictionary mapping subsets S to estimated Fourier coefficients.
        """
        estimated_coeffs = {}
        S = []
        return self.goldreich_levin_recursive(num_samples, 0, epsilon, S, estimated_coeffs)




# Example usage
if __name__ == "__main__":
    def ltf(x):
        # linear threshold function
        if x[0] * 1 + x[1] * 1 + x[2] * 1 + x[3] * 1 + x[4] * 1 < 0:
            return -1
        else:
            return 1
    
    f = binary_function(ltf, 5)

    print(f.fourier_exact())

    print(f.fourier_approximate(10000))

    print(f.goldreich_levin(0.01, 1000))