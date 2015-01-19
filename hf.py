# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012-2013

from __future__ import print_function, division
import numpy as np
import sys
import theano
import theano.tensor as T
import cPickle
import os
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
from itertools import izip


def gauss_newton_product(cost, p, v, s):
    """Computes the product Gv = J'HJv (G is the Gauss-Newton matrix)"""
    Jv = T.Rop(s, p, v)
    HJv = T.grad(T.sum(T.grad(cost, s) * Jv), s,
                 consider_constant=[Jv], disconnected_inputs='ignore')
    Gv = T.grad(T.sum(HJv * s), p, consider_constant=[HJv, Jv],
                disconnected_inputs='ignore')
    Gv = map(T.as_tensor_variable, Gv)  # for CudaNdarray
    return Gv


class HFOptimizer:
    """Black-box Theano-based Hessian-free optimizer.
    See (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011) for details.

    Useful functions:
    __init__ :
        Compiles necessary Theano functions from symbolic expressions.
    train :
    Performs HF optimization following the above references."""

    def __init__(self, p, inputs, s, costs, h=None, ha=None):
        """Constructs and compiles the necessary Theano functions.

        p : list of Theano shared variables
            Parameters of the model to be optimized.
        inputs : list of Theano variables
            Symbolic variables that are inputs to your graph (they should also
            include your model 'output'). Your training examples must fit these.
        s : Theano variable
            Symbolic variable with respect to which the Hessian of the objective is
            positive-definite, implicitly defining the Gauss-Newton matrix. Typically,
            it is the activation of the output layer.
        costs : list of Theano variables
            Monitoring costs, the first of which will be the optimized objective.
        h: Theano variable or None
            Structural damping is applied to this variable (typically the hidden units
            of an RNN).
        ha: Theano variable or None
            Symbolic variable that implicitly defines the Gauss-Newton matrix for the
            structural damping term (typically the activation of the hidden layer). If
            None, it will be set to `h`.
        """

        self.p = p
        self.shapes = [i.get_value().shape for i in p]
        self.sizes = map(np.prod, self.shapes)
        self.positions = np.cumsum([0] + self.sizes)[:-1]
        self.total_size = np.sum(np.array(self.sizes))
        self.mu = None
        self.cg_backtracking = None
        self.cg_dataset = None
        self.preconditioner = None
        self.max_cg_iterations = None
        self.solutions = None
        self.graphical_out = None

        g = T.grad(costs[0], p)
        g = map(T.as_tensor_variable, g)  # for CudaNdarray
        self.f_gc = theano.function(inputs, g + costs, on_unused_input='ignore')  # during gradient computation
        self.f_cost = theano.function(inputs, costs, on_unused_input='ignore')  # for quick cost evaluation

        symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4
        # old code below fails to account for e.g. "row" types
        #v = [symbolic_types[len(i)]() for i in self.shapes]
        v = [i.type() for i in self.p]  # new version
        Gv = gauss_newton_product(costs[0], p, v, s)

        coefficient = T.scalar()  # this is lambda*mu
        if h is not None:  # structural damping with cross-entropy
            h_constant = symbolic_types[h.ndim]()  # T.Rop does not support `consider_constant` yet, so use `givens`
            structural_damping = coefficient * (
                -h_constant * T.log(h + 1e-10) - (1 - h_constant) * T.log((1 - h) + 1e-10)).sum() / h.shape[0]
            if ha is None:
                ha = h
            Gv_damping = gauss_newton_product(structural_damping, p, v, ha)
            Gv = [a + b for a, b in zip(Gv, Gv_damping)]
            givens = {h_constant: h}
        else:
            givens = {}

        self.function_Gv = theano.function(inputs + v + [coefficient], Gv, givens=givens,
                                           on_unused_input='ignore')


    def quick_cost(self, delta=0):
        # quickly evaluate objective (costs[0]) over the CG batch
        # for `current params` + delta
        # delta can be a flat vector or a list (else it is not used)
        if isinstance(delta, np.ndarray):
            delta = self.flat_to_list(delta)

        if type(delta) in (list, tuple):
            for i, d in zip(self.p, delta):
                i.set_value(i.get_value() + d)

        cost = np.mean([self.f_cost(*i)[0] for i in self.cg_dataset.iterate(update=False)])

        if type(delta) in (list, tuple):
            for i, d in zip(self.p, delta):
                i.set_value(i.get_value() - d)

        return cost

    def cg(self, b):
        if self.preconditioner:
            M = self.lambda_ * np.ones_like(b)
            for inputs in self.cg_dataset.iterate(update=False):
                M += self.list_to_flat(self.f_gc(*inputs)[:len(self.p)]) ** 2  # / self.cg_dataset.number_batches**2
            # print 'precond~%.3f,' % (M - self.lambda_).mean(),
            M **= -self.xi
            sys.stdout.flush()
        else:
            M = 1.0

        x = self.cg_last_x if hasattr(self, 'cg_last_x') else np.zeros_like(b)  # sharing information between CG runs
        r = b - self.batch_Gv(x)
        d = M * r
        delta_new = np.dot(r, d)
        phi = []
        backtracking = []
        #backspaces = 0

        for i in xrange(1, 1 + self.max_cg_iterations):
            # adapted from http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (p.51)
            q = self.batch_Gv(d)
            dq = np.dot(d, q)
            # assert dq > 0, 'negative curvature'
            alpha = delta_new / dq
            x = x + alpha * d
            r = r - alpha * q
            s = M * r
            delta_old = delta_new
            delta_new = np.dot(r, s)
            d = s + (delta_new / delta_old) * d

            if i >= int(np.ceil(1.3 ** len(backtracking))):
                backtracking.append((self.quick_cost(x), x.copy(), i))

            phi_i = -0.5 * np.dot(x, r + b)
            phi.append(phi_i)
            if not self.graphical_out:
                progress = ' [CG iter %i, phi=%+.5f, cost=%.5f]' % (i, phi_i, backtracking[-1][0])
                print(progress)
            #sys.stdout.write('\b' * backspaces + progress)
            #sys.stdout.flush()
            #backspaces = len(progress)

            k = max(10, i / 10)
            if i > k and phi_i < 0 and (phi_i - phi[-k - 1]) / phi_i < k * 0.0005:
                break
            if np.isnan(backtracking[-1][0]):  # stop if nan encountered
                break

        self.cg_last_x = x.copy()

        if self.cg_backtracking:
            try:
                # try to pick last non-nan (if 0 is nan, that will be taken
                # care of in the .train iteration part):
                j = np.nanargmin([b[0] for b in backtracking])
            except ValueError:
                j = 0
        else:
            j = len(backtracking) - 1
            while j > 0 and backtracking[j - 1][0] < backtracking[j][0]:
                j -= 1
        if not self.graphical_out:
            print('Backtracked to {}/{}'.format(backtracking[j][2], i))
            sys.stdout.flush()

        return backtracking[j] + (i,)

    def flat_to_list(self, vector):
        return [vector[position:position + size].reshape(shape) for shape, size, position in
                zip(self.shapes, self.sizes, self.positions)]

    @staticmethod
    def list_to_flat(l):
        return np.concatenate([i.flatten() for i in l])

    def batch_Gv(self, vector, lambda_=None):
        v = self.flat_to_list(vector)
        if lambda_ is None:
            lambda_ = self.lambda_
        result = lambda_ * vector  # Tikhonov damping
        for inputs in self.cg_dataset.iterate(False):
            result += self.list_to_flat(
                self.function_Gv(*(inputs + v + [lambda_ * self.mu]))) / self.cg_dataset.number_batches
        return result

    def train(self, gradient_dataset, cg_dataset, initial_lambda=0.1, mu=0.03, cg_backtracking=False,
              preconditioner=True, preconditioner_xi=.75,
              max_cg_iterations=250, num_updates=100, validation=None, validation_frequency=1,
              patience=np.inf, save_progress=None, print_parameters=False,
              graphical_out=False, display_range=100):
        """Performs HF training.

        gradient_dataset : SequenceDataset-like object
            Defines batches used to compute the gradient.
            The `iterate(update=True)` method should yield shuffled training examples
            (tuples of variables matching your graph inputs).
            The same examples MUST be returned between multiple calls to iterator(),
            unless update is True, in which case the next batch should be different.
        cg_dataset : SequenceDataset-like object
            Defines batches used to compute CG iterations.
        initial_lambda : float
            Initial value of the Tikhonov damping coefficient.
        mu : float
            Coefficient for structural damping.
        cg_backtracking : Boolean
            If True, backtracks as much as necessary to find the global minimum among
            all CG iterates. Else, Martens' heuristic is used.
        preconditioner : Boolean
            Whether to use Martens' preconditioner.
            P = (diag(d) + lambda * I)**xi
        preconditioner_xi : 0 < float < 1
        max_cg_iterations : int
            CG stops after this many iterations regardless of the stopping criterion.
        num_updates : int
            Training stops after this many parameter updates regardless of `patience`.
        validation: SequenceDataset object, (lambda : tuple) callback, or None
            If a SequenceDataset object is provided, the training monitoring costs
            will be evaluated on that validation dataset.
            If a callback is provided, it should return a list of validation costs
            for monitoring, the first of which is also used for early stopping.
            If None, no early stopping nor validation monitoring is performed.
        validation_frequency: int
            Validation is performed every `validation_frequency` updates.
        patience: int
            Training stops after `patience` updates without improvement in validation
            cost.
        save_progress: string or None
            A checkpoint is automatically saved at this location after each update.
            Call the `train` function again with the same parameters to resume
            training.
        print_variables: Boolean
            Whether or not print out the variables after each iteration. Default
            is False (keep it that way for big problems!)
        """

        self.lambda_ = initial_lambda
        self.mu = mu
        self.cg_backtracking = cg_backtracking
        self.cg_dataset = cg_dataset
        self.preconditioner = preconditioner
        self.max_cg_iterations = max_cg_iterations
        self.solutions = None
        self.graphical_out = graphical_out
        self.last_good_parameters = None
        self.last_parameters_change = None
        if self.graphical_out:
            self.parameter_store = np.empty((display_range, self.total_size))
            self.parameter_store[:] = np.nan
        best = [0, np.inf, None]  # iteration, cost, params
        first_iteration = 1
        previous_parameters = None
        self.xi = preconditioner_xi

        if isinstance(save_progress, str) and os.path.isfile(save_progress):
            save = cPickle.load(file(save_progress))
            self.cg_last_x, best, self.lambda_, first_iteration, init_p = save
            first_iteration += 1
            for i, j in zip(self.p, init_p):
                i.set_value(j)
            print('* recovered saved model')

        if self.graphical_out:
            f, axarr = plt.subplots(self.total_size, sharex=True,
                                    figsize=(10, 2 * self.total_size))
            axarr[0].set_title('Parameter updates')
            axdata = np.empty(axarr.shape, dtype=object)

        try:
        #TODO: Stopping criterion, max pct change in values < tolerance
            counter_good_values = 0
            for u in xrange(first_iteration, 1 + num_updates):

                gradient = np.zeros(sum(self.sizes), dtype=theano.config.floatX)
                costs = []
                for inputs in gradient_dataset.iterate(update=True):
                    result = self.f_gc(*inputs)
                    gradient += self.list_to_flat(result[:len(self.p)]) /\
                        gradient_dataset.number_batches
                    costs.append(result[len(self.p):])

                if self.graphical_out:
                    print('\rUpdate %i/%i:' % (u, num_updates), end=' ')
                    print('Cost= %.7f,' % (np.mean(costs, axis=0)), end='')
                    print(' lambda={0:.3f}'.format(self.lambda_), end='')
                    sys.stdout.flush()
                else:
                    print('Update %i/%i:' % (u, num_updates), end=' ')
                    print('Cost= %.7f,' % (np.mean(costs, axis=0)), end='')
                    print(' lambda={0:.3f}'.format(self.lambda_))
                    sys.stdout.flush()

                after_cost, flat_delta, backtracking, num_cg_iterations = self.cg(-gradient)
                if np.isnan(after_cost):
                    print('Cost function not real valued. Adjusting lambda and trying again.')
                    self.lambda_ *= 10.
                    continue

                delta_cost = np.dot(flat_delta, gradient + 0.5 *
                                    self.batch_Gv(flat_delta, lambda_=0))  # disable damping
                before_cost = self.quick_cost()
                for i, delta in zip(self.p, self.flat_to_list(flat_delta)):
                    i.set_value(i.get_value() + delta)
                # temp print for low dim cases:
                if print_parameters:
                    print('Values (A, B, b): \n', [i.get_value().copy() for i in self.p])
                cg_dataset.update()

                rho = (after_cost - before_cost) / delta_cost  # Levenberg-Marquardt
                # print 'rho=%f' %rho,
                if rho < 0.25:
                    self.lambda_ *= 1.5
                elif rho > 0.75:
                    self.lambda_ /= 1.5

                if validation is not None and u % validation_frequency == 0:
                    if hasattr(validation, 'iterate'):
                        costs = np.mean([self.f_cost(*i) for i in validation.iterate()], axis=0)
                    elif callable(validation):
                        costs = validation()
                    print('validation=', costs)
                    if costs[0] < best[1]:
                        best = u, costs[0], [i.get_value().copy() for i in self.p]
                        print('*NEW BEST')

                if isinstance(save_progress, str):
                    # do not save dataset states
                    save = self.cg_last_x, best, self.lambda_, u, \
                        [i.get_value().copy() for i in self.p]
                    cPickle.dump(save, file(save_progress, 'wb'), cPickle.HIGHEST_PROTOCOL)

                if u - best[0] > patience:
                    print('PATIENCE ELAPSED, BAILING OUT')
                    break

                current_parameters = np.array([i.get_value().copy()
                                               for i in self.p])
                current_parameters_flat = np.hstack([i.get_value().copy().flatten()
                                               for i in self.p])

                if np.prod(~np.isnan(current_parameters_flat)): #  check that no nans in parameters
                    counter_good_values += 1
                    if previous_parameters is not None:
                        parameters_change = [(x - y) / y for x, y in
                                             izip(current_parameters, previous_parameters)]
                        self.last_parameters_change = parameters_change
                        self.solutions = previous_parameters  # NOT current because they may be bad
                    previous_parameters = current_parameters



                if self.graphical_out:
                    #parameter_plot = plt.plot(parameter_store)
                    if u <= display_range:
                        self.parameter_store[u - 1] = current_parameters_flat
                    else:
                        self.parameter_store = np.roll(self.parameter_store, -1, axis=0)
                        self.parameter_store[-1] = current_parameters_flat

                    #parameter_plot.set_data(parameter_store)
                    #plt.draw()


                    for n in range(self.total_size):
                        try:
                            axarr[n].lines.remove(axdata[n][0])
                        except:
                            pass
                        axdata[n] = axarr[n].plot(
                            range(display_range), self.parameter_store[:, n], 'b')
                        #axarr[n].set_autoscaley_on(True)
                        minval = np.nanmin(axdata[n][0].get_data()[1])
                        maxval = np.nanmax(axdata[n][0].get_data()[1])
                        axarr[n].set_ylim(minval, maxval)
                    display.clear_output(wait=True)
                    display.display(pl.gcf())
                    #plt.show(block=False)
                    #plt.draw()

                    #print('\n', parameter_store)


                print
                sys.stdout.flush()
        except KeyboardInterrupt:
            print('\nInterrupted by user.')

        if self.solutions is None:
            print('\nSolutions stored as .solutions.')
            print('Last good values obtained at iteration {}.'.format(counter_good_values))
        #return best[2]

#TODO: replace SequenceDataset with Pandas DataFrame/ HDFStore!
class SequenceDataset:
    """Slices, shuffles and manages a small dataset for the HF optimizer."""

    def __init__(self, data, batch_size, number_batches, minimum_size=10):
        """SequenceDataset __init__

        data : list of lists of numpy arrays
            Your dataset will be provided as a list (one list for each graph input) of
            variable-length tensors that will be used as mini-batches. Typically, each
            tensor is a sequence or a set of examples.
        batch_size : int or None
            If an int, the mini-batches will be further split in chunks of length
            `batch_size`. This is useful for slicing subsequences or provide the full
            dataset in a single tensor to be split here. All tensors in `data` must
            then have the same leading dimension.
        number_batches : int
            Number of mini-batches over which you iterate to compute a gradient or
            Gauss-Newton matrix product.
        minimum_size : int
            Reject all mini-batches that end up smaller than this length.
        """
        self.current_batch = 0
        self.number_batches = number_batches
        self.items = []

        for i_sequence in xrange(len(data[0])):
            if batch_size is None:
                self.items.append([data[i][i_sequence] for i in xrange(len(data))])
            else:
                for i_step in xrange(0, len(data[0][i_sequence]) - minimum_size + 1, batch_size):
                    self.items.append([data[i][i_sequence][i_step:i_step + batch_size] for i in xrange(len(data))])

        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.items)

    def iterate(self, update=True):
        for b in xrange(self.number_batches):
            yield self.items[(self.current_batch + b) % len(self.items)]
        if update:
            self.update()

    def update(self):
        if self.current_batch + self.number_batches >= len(self.items):
            self.shuffle()
            self.current_batch = 0
        else:
            self.current_batch += self.number_batches