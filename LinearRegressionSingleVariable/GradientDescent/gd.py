import numpy
import argparse

def linear_regression_normal_equation(points, starting_b, starting_m):
    mean_array = numpy.mean(points, axis=0)
    mean_x = mean_array[0]
    mean_y = mean_array[1]

    x_mean_err = numpy.subtract(points[:,0], mean_x)
    x_mean_sqr_error = numpy.square(x_mean_err)
    y_mean_err = numpy.subtract(points[:,1], mean_y)

    m = numpy.sum(numpy.multiply(x_mean_err, y_mean_err))/numpy.sum(x_mean_sqr_error)
    b = mean_y - (m * mean_x)
    return [b, m]

def RSS(b, m, points, print_rss):
    rss = 0.0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        rss += (y - (m*x + b)) ** 2
    if print_rss:
        print("RSS: {0:.2f}".format(rss))
    return rss

def compute_error_for_line_given_points(b, m, points):
    rss = RSS(b, m, points, False)    
    return rss / len(points)

def step_gradient(b_current, m_current, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float((len(points)))
    for i in range(int(N)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m, b_gradient, m_gradient]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, epsilon, print_rss, print_gradient, normal_equation):
    b = starting_b
    m = starting_m
    iter_count = 0
    out_f = None
    if normal_equation:
        iter_count = 1
        [b, m] = linear_regression_normal_equation(numpy.array(points), starting_b, starting_m)
    else:
        if print_gradient:
            out_f = open("gradients.csv","w")
        for i in range(num_iterations):
            # print("#iter: {:d}".format(i))
            [b, m, b_gradient, m_gradient] = step_gradient(b, m, numpy.array(points), learning_rate)
            if print_gradient:
                out_f.write("{:d},{:.5f},{:.5f}\n".format((i+1), b_gradient, m_gradient))
            rss = RSS(b, m, points, print_rss)
            iter_count += 1
            if abs(b_gradient) < epsilon and abs(m_gradient) < epsilon:
                # print("b_gradient: {:.5f}; m_gradient: {:.5f}; epsilon: {:.5f}".format(b_gradient, m_gradient, epsilon))
                break
    if out_f is not None:
        out_f.close()
    return [b, m, iter_count]

def run(args):
    points = None
    try:
        points = numpy.genfromtxt(args.data_file, delimiter=",")
        if len(points[0]) == 1:
            raise Exception('')
    except Exception:
        points = numpy.genfromtxt(args.data_file, delimiter=" ")
        if len(points[0]) == 1:
            raise Exception('Unexpected file delimiter!')
            
    # hyperparameters
    learning_rate = args.learning_rate

    # y = mx + b (slope formula)
    initial_b = args.initial_b
    initial_m = args.initial_m

    print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m, num_iterations_needed] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, args.num_iterations, args.epsilon, args.print_rss, args.print_gradient, args.normal_equation)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations_needed, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default="data.csv", dest='data_file', type=str)
    parser.add_argument('-b', '--initial-b', default=0.0, dest='initial_b', type=float)
    parser.add_argument('-m', '--initial-m', default=0.0, dest='initial_m', type=float)
    parser.add_argument('-i', '--num-iterations', default=1000, dest='num_iterations', type=int)
    parser.add_argument('-r', '--learning-rate', default=0.0001, dest='learning_rate', type=float)
    parser.add_argument('-e', '--min-gradient', default=0.00, dest='epsilon', type=float)
    parser.add_argument('--show-rss', action='store_true', dest='print_rss')
    parser.add_argument('--show-gradient', action='store_true', dest='print_gradient')
    parser.add_argument('--normal-equation', action='store_true', dest='normal_equation')
    args = parser.parse_args()

    run(args)