from importlib.metadata import distribution
import numpy as np
import itertools
import scipy
import qiskit
import matplotlib.pyplot as plt

from typing import Union
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.quantum_info import state_fidelity
from qiskit_experiments.library import StateTomography
from qiskit.circuit.library import StatePreparation 
from qiskit.opflow import (
    CircuitSampler,
    ExpectationFactory,
    CircuitStateFn,
    StateFn,
    I, X, Y, Z
)

from scipy.optimize import minimize

class VQT:
    def __init__(self, 
        hamiltonian: qiskit.opflow.OperatorBase, 
        beta: float,
        ansatz: qiskit.QuantumCircuit,  
        backend: Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance]
    ) -> None:
        """Variational Quantum Thermalizer Implementation.

        Args:
            hamiltonian (qiskit.opflow.OperatorBase): Hamiltonian that you want to get 
            the expectation value.
            beta (float): inverse temperature.
            ansatz (qiskit.QuantumCircuit): QuantumCircuit for the ansatz.
            backend (Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance]): Backend
            that you want to run.
        """

        self.hamiltonian = hamiltonian
        self.beta = beta
        self.ansatz = ansatz
        self.backend = backend

        self.num_qubits = ansatz.num_qubits
        

    def calculate_entropy(self, distribution: list) -> np.array:
        """Calculates the entropy for a given distribution.
        Returns an array of the entropy values of the different initial density matrices.

        Args:
            distribution (list): probability distribution

        Returns:
            np.array: entropy values of the different initial density matrices
        """
        total_entropy = 0
        for d in distribution:
            total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])

        return total_entropy

    def convert_list(self, params: np.array, n_rotations: int=3) -> tuple:
        """Converts the list of parameters of the ansatz and
        split them into the distribution parameters and 
        the ansatz parameters.

        Args:
            params (np.array): list of ansatz params.
            n_rotations (int, optional): number of rotation parameters.
            per layer. Defaults to 3.

        Returns:
            tuple: tuple containing two lists, one with the distribution 
            parameters and the other with the ansatz parameters.
        """

        # Separates the list of parameters
        dist_params = params[0:self.num_qubits]
        ansatz_params = params[self.num_qubits:]

        return dist_params, ansatz_params

    def sample_ansatz(self,
        ansatz: qiskit.QuantumCircuit,
        ansatz_params: list, 
        ) -> float:
        """Samples a hamiltonian given an ansatz, which is a Quantum circuit
        and outputs the expected value given the hamiltonian.

        Args:
            ansatz (qiskit.QuantumCircuit): Quantum circuit that you want to get the expectation
            value.
            ansatz_params (list): List of parameters of the ansatz.

        Returns:
            float: Expectation value.
        """

        if qiskit.utils.quantum_instance.QuantumInstance == type(self.backend):
            sampler = CircuitSampler(self.backend, param_qobj=is_aer_provider(self.backend.backend))
        else:
            sampler = CircuitSampler(self.backend)

        expectation = ExpectationFactory.build(operator=self.hamiltonian, backend=self.backend)
        observable_meas = expectation.convert(StateFn(self.hamiltonian, is_measurement=True))

        ansatz = ansatz.bind_parameters(ansatz_params)

        ansatz_circuit_op = CircuitStateFn(ansatz)

        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        sampled_expect_op = sampler.convert(expect_op)

        return np.real(sampled_expect_op.eval())

    
    def ansatz_list(self) -> list:
        """Generates a list of the ansatzes with all combination of inital states.

        Returns:
            list: list with ansatzes and all combinations of initial states.
        """

        list_circs = []

        combos = [''.join(bs) for bs in itertools.product('01', repeat=self.num_qubits)]

        for comb in combos:
            qc_init = qiskit.QuantumCircuit(self.num_qubits)
            
            qc_init.append(StatePreparation(comb), qc_init.qubits)

            qc_init.append(self.ansatz, qc_init.qubits)

            list_circs.append(qc_init)

        return list_circs
    

    def exact_cost(self, params: list) -> float:
        """Calculates the exact cost of the ansatz.

        Args:
            params (list): list contaning the parameters for the 
            distribution and the ansatz.

        Returns:
            float: calculated cost function.
        """

        # Transforms the parameter list
        dist_params, ansatz_params = self.convert_list(params)

        # Creates the probability distribution
        distribution = prob_dist(dist_params)

        # Generates a list of all computational basis states of the 
        # qubit system
        combos = itertools.product([0, 1], repeat=self.num_qubits)
        s = [list(c) for c in combos]

        ansatzes = self.ansatz_list()

        # Passes each basis state through the variational circuit 
        # and multiplies the calculated energy EV with the associated 
        # probability from the distribution
        cost = 0
        for i, ansatz in zip(s, ansatzes):
            result = self.sample_ansatz(ansatz,
                                        ansatz_params
                                       )
            for j in range(0, len(i)):
                result = result * distribution[j][i[j]]
            cost += result

        # Calculates the entropy and the final cost function
        entropy = self.calculate_entropy(distribution)
        final_cost = self.beta * cost - entropy

        return final_cost
    
    
    def cost_execution(self, params):
        """Executes the cost step, counts the number of iterations,
        and appends the cost history to a list.

        Args:
            params (list): list contaning the parameters for the 
            distribution and the ansatz. 

        Returns:
            float: calculated cost function.
        """
        
        cost = self.exact_cost(params)

        self.history.append(float(cost))

        if self.iterations % 5 == 0:
            print("Cost at Step {}: {}".format(self.iterations, cost))

        self.iterations += 1
        return cost
    

    def vqt_optimization(self, optimizer='COBYLA', optimizer_config={"maxiter": 10}, 
    random_seed=42, plot=True, plot_color='tab:blue'):
        """Performs the classical optimization of the cost function.

        Args:
            num_qubits (int): number of qubits.
            optimizer (str): classical optimization method. Defaults to 'COBYLA'.
            optimizer_config (dict, optional): dict with optimizer configurations.
            random_seed (int, optional): random seed for reproducibility. Defaults to 42.
            plot (bool, optional): If True, show cost plot. Defaults to True.
            plot_color (str, optional): Set the color of the plot. Defaults to 'tab:blue'.

        Returns:
            list: The parameters of the probability distribution and of the ansatz.
        """

        self.iterations = 0
        self.history = []
        
        np.random.seed(random_seed)

        number = self.num_qubits + self.ansatz.num_parameters
        params = [np.random.randint(-300, 300) / 100 for i in range(0, number)]
        
        print("Training...")
        
        out = minimize(self.cost_execution, 
                       x0=params,
                       method=optimizer, 
                       options=optimizer_config
                       )
        
        print("Finished after " + str(self.iterations) + " steps.")

        if plot == True:
            self.plot_training(plot_color)
    
        return out["x"], self.history
    

    def best_circuits(self, ansatz_params: list) -> list:
        """Prepares a list with the 2 ** num_qubits circuits parameterized with 
        the optimal parameters.

        Args:
            ansatz_params (list): list containing the optimal parameters.

        Returns:
            list: a list containing the optimal circuits.
        """

        ansatzes = self.ansatz_list()

        circuit_list = []

        for circ in ansatzes:

            circ = circ.bind_parameters(ansatz_params)
            
            circuit_list.append(circ)

        return circuit_list


    def run_vqt(self, optimizer='COBYLA', optimizer_config={"maxiter": 10}, random_seed=42, 
                plot=True, plot_color='tab:blue') -> dict:
        """Runs the optimiztion and returns the optimized parameters, the history and parameterized
        circuits.

        Args:
            num_qubits (int): number of qubits.
            optimizer (str): classical optimization method. Defaults to 'COBYLA'.
            optimizer_config (dict, optional): dict with optimizer configurations.
            random_seed (int, optional): random seed for reproducibility. Defaults to 42.
            plot (bool, optional): If True, show cost plot. Defaults to True.
            plot_color (str, optional): Set the color of the plot. Defaults to 'tab:blue'.

        Returns:
            dict: Dictionary with results
        """

        best_params, history = self.vqt_optimization(optimizer, optimizer_config, 
                               random_seed, 
                               plot, plot_color
                               )

        dist_params, ansatz_params = self.convert_list(best_params)

        best_circuits = self.best_circuits(ansatz_params)

        self.results = {"ansatz_params": ansatz_params,
                        "dist_params": dist_params,
                        "prob_dist": prob_dist(dist_params),
                        "best_circuits": best_circuits,
                        "history": history,                             
                       }

        return self.results


    def get_results(self) -> dict:
        """Return results.

        Returns:
            dict: Dictionary with results
        """
        try:
            return self.results

        except:
            print("Results not available yet. Call run_vqt() method to obtain results.")              


    def plot_training(self, plot_color):
        plt.plot(range(1, len(self.history)+1), self.history, '.-', color=plot_color)
        #plt.xticks(range(1,len(self.history)+1))
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
    
    
    def prepare_state(self) -> np.array:
        """Prepares the thermal state density matrix by a weighted sum of 
        the ensemble's pure states and their respective probabilities.

        Returns:
            np.array: Density matrix of the thermal state.
        """        

        # Performs the quantum state tomography:
        states = tomography(backend = self.backend,
                            circuits = self.results['best_circuits']
                           )

        dist_final = self.results['prob_dist']

        # Initializes the density matrix

        final_density_matrix = np.zeros((2 ** self.num_qubits, 2 ** self.num_qubits))

        combos = itertools.product([0, 1], repeat=self.num_qubits)
        s = [list(c) for c in combos]

        for i, sample in zip(s, states):

            state = sample.analysis_results("state").value

            dist_aux = 1
            for j in range(0, len(i)):
                dist_aux = dist_final[j][i[j]]*dist_aux
            final_density_matrix = np.add(final_density_matrix, dist_aux*state)

        return np.array(final_density_matrix)
    

    def get_density_matrix(self, fidelity = False) -> Union[np.array, tuple]:
        """Returns the final density matrix.

        Args:
            fidelity (bool, optional): Whether or not to calculate state fidelity. 
            Defaults to False.

        Returns:
            np.array: Final density matrix.
        """

        final_density_matrix = self.prepare_state()

        if fidelity == False:
            self.density_matrix = final_density_matrix

            return final_density_matrix

        else:
            target_matrix = create_target(self.beta, self.hamiltonian.to_matrix())
            fid = state_fidelity(final_density_matrix, target_matrix)

            self.density_matrix = final_density_matrix
            self.fidelity = fid

            return (final_density_matrix, fid)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
        x (float): function argument.

    Returns:
        float: evaluated sigmoid function.
    """
    
    return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params: list) -> np.array:
    """Calculates the probability distribution of the sigmoid function.

    Args:
        params (list): list of probability distribution parameters.

    Returns:
        np.array: return array with stacked parameters. 
    """
    return np.vstack([1 - sigmoid(params), sigmoid(params)]).T

def tomography(backend: Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance], 
               circuits: list
               ) -> list:
    """Performs Quantum State Tomography for all parameterized VQT circuits.  

    Args:
        backend (Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance]): 
        Backend to run the circuit
        circuits (list): list contaning all the circuits to be tomographed.

    Returns:
        list: a list containing all the matrices 
    """

    states = []

    for circ in circuits:
        qst_exp = StateTomography(circ)
        qst_data = qst_exp.run(backend, seed_simulation=100).block_for_results()

        states.append(qst_data)

    return states

def create_target(beta: float, hamiltonian: np.array) -> np.array:
    """Calculates the matrix form of the density matrix, by taking
     the exponential of the Hamiltonian.

    """

    y = -1 * float(beta) * hamiltonian
    new_matrix = scipy.linalg.expm(np.array(y))
    norm = np.trace(new_matrix)
    final_target = (1 / norm) * new_matrix

    return final_target