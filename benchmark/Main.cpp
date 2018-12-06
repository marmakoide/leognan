#include <cstdlib>
#include <iostream>
#include <sstream>

#include <list>
#include <vector>

#include <leognan.h>
#include <leognan/Macros>

using namespace std;
using namespace leognan;



// --- Initial solutions generator --------------------------------------------

class InitialSolutionGenerator {
public:
	InitialSolutionGenerator(double a, double b, int n) {
		m_center = Eigen::VectorXd::Constant(n, (a + b) / 2);
		m_extent = Eigen::VectorXd::Constant(n, std::fabs(a - b));
	}

	InitialSolutionGenerator(const Eigen::VectorXd& A,
	                         const Eigen::VectorXd& B) {
		Eigen::VectorXd min_corner = A.array().min(B.array());
		Eigen::VectorXd max_corner = A.array().max(B.array());

		m_center = (max_corner + min_corner) / 2;
		m_extent = (max_corner - min_corner);
	}

	InitialSolutionGenerator(const InitialSolutionGenerator& other) :
		m_center(other.m_center),
		m_extent(other.m_extent) { }

	Eigen::VectorXd
	get(Randomizer& rnd) const {
		Eigen::VectorXd ret(m_extent.rows());
		for(Eigen::Index i = 0; i < m_extent.rows(); ++i)
			ret(i) = rnd.next_uniform();

		ret.array() -= .5;
		ret.array() *= m_extent.array();
		ret += m_center;
		return ret;
	}

private:
	Eigen::VectorXd m_center;
	Eigen::VectorXd m_extent;
}; // class InitialSolutionGenerator



// --- Fitness functions ------------------------------------------------------

class QuadraticFunction : public FitnessFunction {
public:
	QuadraticFunction(const Eigen::VectorXd& A,
	                  const Eigen::MatrixXd& M) :
		mA(A),
		mM(M) { }

	virtual ~QuadraticFunction() { }

	virtual int
	get_dimension() const {
		return mA.rows();
	}

	virtual double
	get_fitness(const Eigen::VectorXd& X) const {
		return (mA.asDiagonal() * (mM * X)).squaredNorm();
	}

private:
	Eigen::VectorXd mA;
	Eigen::MatrixXd mM;
}; // class QuadraticFunction



class RastriginFunction : public FitnessFunction {
public:
	RastriginFunction(int n,
	                  const Eigen::MatrixXd& M) :
		m_n(n),
		m_M(M) { }

	virtual int
	get_dimension() const {
		return m_n;
	}

	virtual double
	get_fitness(const Eigen::VectorXd& X) const {
		return 10. * m_n + (X.array().square() - 10. * ((2 * M_PI) * m_M * X).array().cos()).sum();
	}

private:
	int m_n;
	Eigen::MatrixXd m_M;
}; // class RastriginFunction



// --- Fitness function factory -----------------------------------------------

Eigen::MatrixXd
get_random_rotation_matrix(int n, Randomizer& rnd) {
	// Fill a matrix with normally distributed values
	Eigen::MatrixXd A(n, n);
	for(Eigen::Index i = 0; i < n; ++i)
		for(Eigen::Index j = 0; j < n; ++j)
			A(i, j) = rnd.next_gaussian(1.);

	// Orthogonalize the matrix
	Eigen::HouseholderQR<Eigen::MatrixXd> QR(A);
	Eigen::MatrixXd B(n, n);
  B.setIdentity();
  B = QR.householderQ() * B;	

	// Job done
	return B;
}



class FitnessFunctionFactory {
public:
	virtual ~FitnessFunctionFactory() { }

	virtual FitnessFunction*
	get_fitness_function(Randomizer& rnd) const = 0;
}; // class FitnessFunctionFactory



class QuadraticFunctionFactory : public FitnessFunctionFactory {
public:
	QuadraticFunctionFactory(const Eigen::VectorXd& A) :
		mA(A) { }

	virtual ~QuadraticFunctionFactory() { }

	virtual FitnessFunction*
	get_fitness_function(Randomizer& rnd) const {
		return new QuadraticFunction(mA, get_random_rotation_matrix(mA.rows(), rnd));
	}

private:
	Eigen::VectorXd mA;
}; // class QuadraticFunctionFactory



class RastriginFunctionFactory : public FitnessFunctionFactory {
public:
	RastriginFunctionFactory(int n) :
		m_n(n) { }

	virtual ~RastriginFunctionFactory() { }

	virtual FitnessFunction*
	get_fitness_function(Randomizer& rnd) const {
		return new RastriginFunction(m_n, get_random_rotation_matrix(m_n, rnd));
	}

private:
	int m_n;
}; // class RastriginFunctionFactory



// --- Test cases -------------------------------------------------------------

class TestCase {
public:
	TestCase(const std::string& name,
	         const FitnessFunctionFactory* fitness_function_factory,
	         double sigma_init,
	         const InitialSolutionGenerator& ini_gen) :
		m_name(name),
		m_fitness_function_factory(fitness_function_factory),
		m_sigma_init(sigma_init),
		m_ini_gen(ini_gen) { }

	inline const std::string& name() const {
		return m_name;
	}

	inline const FitnessFunctionFactory* fitness_function_factory() const {
		return m_fitness_function_factory;
	}

	inline double sigma_init() const { 
		return m_sigma_init;
	}

	inline const InitialSolutionGenerator& initial_solution_generator() const {
		return m_ini_gen;
	}

private:
	const std::string m_name;
	const FitnessFunctionFactory* m_fitness_function_factory;
	double m_sigma_init;
	InitialSolutionGenerator m_ini_gen;
}; // class TestCase



TestCase
get_sphere_test(int n) {
	Eigen::VectorXd A = Eigen::VectorXd::Ones(n);

	// Generate the name
	std::ostringstream name;
	name << "sphere-" << n;

	// Job done
	return
		TestCase(name.str(),
		         new QuadraticFunctionFactory(A),
		         3.,
		         InitialSolutionGenerator(-1., 5., n));
}



TestCase
get_ellipsoid_test(int n) {
	Eigen::VectorXd A(n);
	for(int i = 0; i < n; ++i)
		A(i) = std::pow(1000., double(i) / (n - 1));

	// Generate the name
	std::ostringstream name;
	name << "ellipsoid-" << n;

	// Job done
	return
		TestCase(name.str(),
		         new QuadraticFunctionFactory(A),
		         3.,
		         InitialSolutionGenerator(-1., 5., n));
}



TestCase
get_tablet_test(int n) {
	Eigen::VectorXd A = Eigen::VectorXd::Ones(n);
	A(0) = 1000.;
	
	// Generate the name
	std::ostringstream name;
	name << "tablet-" << n;

	// Job done
	return
		TestCase(name.str(),
		         new QuadraticFunctionFactory(A),
		         3.,
		         InitialSolutionGenerator(-1., 5., n));
}



TestCase
get_rastrigin_test(int n) {
	// Generate the name
	std::ostringstream name;
	name << "rastrigin-" << n;

	// Job done
	return
		TestCase(name.str(),
		         new RastriginFunctionFactory(n),
		         3.,
		         InitialSolutionGenerator(-1., 5., n));
}



// --- Testing routine --------------------------------------------------------

std::vector<double>
run_optimization(const FitnessFunction* fitness_func,
                 std::size_t epoch_count,
		             Randomizer& rnd,
		             const Eigen::VectorXd& X_init) {
	std::vector<double> fitness_log;

	// Initialize the optimizer
	OnePlusOneCMAES optim;
	optim.init(fitness_func, X_init);
	fitness_log.push_back(optim.get_best_fitness());

	// Run the optimizer
	for(std::size_t i = 0; i < epoch_count; ++i) {
		optim.update(rnd, fitness_func);
		fitness_log.push_back(optim.get_best_fitness());
		if (optim.get_best_fitness() < 1e-10)
			break;
	}

	// Job done
	return fitness_log;
}



void
run_test(const TestCase& test,
		     std::size_t trial_count,
         std::size_t epoch_count,
         Randomizer& rnd) {
	// Accumulate runs to make statistics
	std::list< std::vector<double> > fitness_stat;
	for(std::size_t i = 0; i < trial_count; ++i) {
		Eigen::VectorXd X_init = test.initial_solution_generator().get(rnd);
		FitnessFunction* fitness_func = test.fitness_function_factory()->get_fitness_function(rnd);
		std::vector<double> fitness_log = run_optimization(fitness_func, epoch_count, rnd, X_init);
		delete fitness_func;
		fitness_stat.push_back(fitness_log);
	}

	// Compute median fitness statistics
	std::size_t max_stat_len = 0;
	for(const auto& fitness_list : fitness_stat)
		max_stat_len = std::max(max_stat_len, fitness_list.size());

	std::vector<double> median_fitness;
	for(std::size_t i = 0; i < max_stat_len; ++i) {
		std::vector<double> value_list;
		for(const auto& fitness_list : fitness_stat) {
			if (i >= fitness_list.size())
				value_list.push_back(fitness_list.back());
			else
				value_list.push_back(fitness_list[i]);
		}

		// Compute the median
		std::sort(value_list.begin(), value_list.end());
		if (value_list.size() % 2 == 0)
			median_fitness.push_back(.5 * (value_list[value_list.size() / 2] + value_list[1 + value_list.size() / 2]));
		else
			median_fitness.push_back(value_list[value_list.size() / 2]);
	}

	// Output result
	cout << "# " << test.name() << endl;
	for(auto value : median_fitness)
		cout << value << endl;
}



// --- Main entry point -------------------------------------------------------

int
main(int UNUSED_PARAM(argc), char** UNUSED_PARAM(argv)) {
	// Initialize the randomizer
	Randomizer rnd;
	rnd.seed();	

	// Build the test suite
	std::vector<TestCase> test_case_list;

	test_case_list.push_back(get_sphere_test(5));
	test_case_list.push_back(get_ellipsoid_test(5));
	test_case_list.push_back(get_rastrigin_test(5));

	test_case_list.push_back(get_sphere_test(20));
	test_case_list.push_back(get_ellipsoid_test(20));
	test_case_list.push_back(get_rastrigin_test(20));

	// Run test suite
	for(const TestCase& test_case : test_case_list)
		run_test(test_case, 51, 50000, rnd);

	// Job done
	return EXIT_SUCCESS;
}


