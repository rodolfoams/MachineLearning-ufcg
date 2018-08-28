from numpy import *
import argparse

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in xrange(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float((len(points)))
    for i in xrange(int(N)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in xrange(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]

def run(args):
    points = None
    try:
        points = genfromtxt(args.data_file, delimiter=",")
        if len(points[0]) == 1:
            raise Exception('')
    except Exception:
        points = genfromtxt(args.data_file, delimiter=" ")
        if len(points[0]) == 1:
            raise Exception('Unexpected file delimiter!')
            
    # hyperparameters
    learning_rate = args.learning_rate

    # y = mx + b (slope formula)
    initial_b = args.initial_b
    initial_m = args.initial_m

    num_iterations = args.num_iterations

    print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)) )
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default="data.csv", dest='data_file', type=str)
    parser.add_argument('-b', '--initial-b', default=0.0, dest='initial_b', type=float)
    parser.add_argument('-m', '--initial-m', default=0.0, dest='initial_m', type=float)
    parser.add_argument('-i', '--num-iterations', default=1000, dest='num_iterations', type=int)
    parser.add_argument('-r', '--learning-rate', default=0.0001, dest='learning_rate', type=float)
    args = parser.parse_args()

    run(args)